# cython: profile=True
# cython: infer_types=True
'''
Created on Jul 24, 2012

@author: Nathan Schneider (nschneid)
'''
from __future__ import print_function, division
import sys, codecs, random, operator, math
from collections import defaultdict, Counter

cimport cython
from cython.view cimport array as cvarray

from labeledSentence import LabeledSentence
import morph
from pyutil.memoize import memoize
from pyutil.ds import features 

import supersenseFeatureExtractor
featureExtractor = None


from dataFeaturizer import SupersenseDataSet, SupersenseFeaturizer



@cython.profile(False)
cdef inline int _ground0(int liftedFeatureIndex, int labelIndex, int numFeatures):
    return liftedFeatureIndex + labelIndex*numFeatures

cdef inline int _ground(int liftedFeatureIndex, int labelIndex, object indexer):
    return _ground0(liftedFeatureIndex, labelIndex, len(indexer))

cdef float _score(object featureMap, float[:] weights, int labelIndex, int indexerSize):
        '''Compute the dot product of a set of feature values and the corresponding weights.'''
        if labelIndex==-1:
            return 0.0
        
        dotProduct = 0.0
        for h,v in featureMap.items():
            dotProduct += weights[_ground0(h, labelIndex, indexerSize)]*v
        return dotProduct

cdef float l2norm(weights):
    cdef float t
    t = 0.0
    for w in weights:
        t += w*w
    return t**0.5

#cdef float _scoreBound(float[:] weights, ):

I_BAR, I_TILDE, i_BAR, i_TILDE = 'ĪĨīĩ'.decode('utf-8')

@memoize
def legalTagBigram(lbl1, lbl2, useBIO=False):
        '''
        For use in decoding. If useBIO is true, valid bigrams include
          B        I
          B-class1 I-class1
          I-class1 I-class1
          O        B-class1
          I-class1 O
        and invalid bigrams include
          B-class1 I-class2
          O        I-class2
          O        I
          B        I-class2
        where 'class1' and 'class2' are names of chunk classes.
        If useBIO is false, no constraint is applied--all tag bigrams are 
        considered legal.
        For the first token in the sequence, lbl1 should be null.
        '''
        if not useBIO: return True
        
        assert lbl2 is None or lbl2[0] in {'O','B','I','o','b','i',I_BAR,I_TILDE,i_BAR,i_TILDE}
        assert lbl1 is None or lbl1[0] in {'O','B','I','o','b','i',I_BAR,I_TILDE,i_BAR,i_TILDE}
        
        if lbl2 is None:
            assert lbl1 is not None
            if lbl1[0] not in {'O', 'I', I_BAR, I_TILDE}:
                if lbl1[0]!='B' or useBIO=='NO_SINGLETON_B':
                    return False
        elif lbl2[0] in {'o', 'b', 'I', I_BAR, I_TILDE}:
            if lbl1 is None or lbl1[0]=='O':
                return False
            elif lbl1[0]=='b' and useBIO=='NO_SINGLETON_B':
                return False
            
            if lbl2[0] not in {'o', 'b'} and (len(lbl1)>1)!=(len(lbl2)>1):
                return False    # only allow I without class if previous tag has no class
            if len(lbl2)>1 and lbl1[2:]!=lbl2[2:]:
                return False    # disallow an I tag following a tag with a different class
        elif lbl2[0] in {'i', i_BAR, i_TILDE}:
            if lbl1 is None or lbl1[0] not in {'b', 'i', i_BAR, i_TILDE}:
                return False
        elif lbl2[0] in {'O', 'B'}:
            if lbl1 is not None and lbl1[0] not in {'O', 'I', I_BAR, I_TILDE}:
                if lbl1[0]!='B' or useBIO=='NO_SINGLETON_B':
                    return False
        return True

cdef c_viterbi(sent, o0Feats, float[:] weights, 
              float[:, :] dpValues, int[:, :] dpBackPointers, 
              labels, featureIndexes, includeLossTerm=False, costAugVal=0.0, useBIO=False):
        '''Uses the Viterbi algorithm to decode, i.e. find the best labels for the sequence 
        under the current weight vector. Updates the predicted labels in 'sent'. 
        Used in both training and testing.'''
        
        indexerSize = len(featureIndexes)
        
        hasFOF = featureExtractor.hasFirstOrderFeatures()
        
        cdef int nTokens, i, k, l, maxIndex
        cdef float score, score0, maxScore, NEGINF
        
        NEGINF = float('-inf')
        nTokens = len(sent)
        nLabels = len(labels)
        
        o1FeatWeights = {l: {} for l in range(nLabels)}   # {current label -> {prev label -> weight}}
        
        prevLabel = None
        
        for i, tok in enumerate(sent):
            sent[i] = tok._replace(prediction=None)
        
        for i in range(nTokens):
            #o0FeatureMap = featureExtractor.extractFeatureValues(sent, i, usePredictedLabels=True, orders={0}, indexer=self._featureIndexes)
            o0FeatureMap = o0Feats[i]
            
            for l,label in enumerate(labels):
                
                # initialize stuff
                maxScore = NEGINF
                maxIndex = -1
                
                # score for zero-order features
                score0 = _score(o0FeatureMap, weights, l, indexerSize)
                
                # cost-augmented decoding
                if label!=sent[i].gold:
                    if includeLossTerm:
                        score0 += 1.0   # base cost of any error
                    if label=='O' and sent[i].gold=='B':
                        score0 += costAugVal    # recall-oriented penalty (for erroneously predicting 'O')
                
                if i==nTokens-1 and not legalTagBigram(label, None, useBIO):
                    maxScore = score = NEGINF
                elif i==0:
                    score = score0
                    if not legalTagBigram(None, label, useBIO):
                        score = NEGINF
                    maxScore = score
                    maxIndex = 0    # doesn't matter--start of sequence
                    # not scoring a start-of-sequence bigram (or end-of-sequence, for that matter).
                    # the beginning/end of the sentence is easily captured with zero-order features.
                else:
                    # consider each possible previous label
                    for k,prevLabel in enumerate(labels):
                        if not legalTagBigram(prevLabel, label, useBIO):
                            continue
                        
                        # compute correct score based on previous scores
                        score = dpValues[i-1,k]
                        
                        # the score for the previou label is added on separately here,
                        # in order to avoid computing the whole score--which only 
                        # depends on the previous label for one feature--a quadratic 
                        # number of times
                        # TODO: plus vs. times doesn't matter here, right? use plus to avoid numeric overflow
                        
                        # score of moving from label k at the previous position to the current position (i) and label (l)
                        score += score0
                        if hasFOF:
                            '''
                            o1FeatureMap = featureExtractor.extractFeatureValues(sent, i, usePredictedLabels=True, orders={1}, indexer=self._featureIndexes)
                            for h,v in o1FeatureMap.items():
                                score += weights[self.getGroundedFeatureIndex(h, l)]*v
                            '''
                            # TODO: generalize this to allow other kinds of first-order features?
                            if k not in o1FeatWeights[l]:
                                o1FeatWeights[l][k] = weights[_ground0(featureIndexes[('prevLabel=',prevLabel)], l, indexerSize)]
                            score += o1FeatWeights[l][k]
                            
                        # find the max of the combined score at the current position
                        # and store the backpointer accordingly
                        if score>maxScore:
                            maxScore = score
                            maxIndex = k
                    
                dpValues[i,l] = maxScore
                #assert maxIndex>=0,(maxIndex,maxScore,i,nTokens,label) # BAD ASSERTION
                dpBackPointers[i,l] = maxIndex
        
        # decode from the lattice
        # extract predictions from backpointers
        
        # first, find the best label for the last token
        maxIndex, maxScore = max(enumerate(dpValues[nTokens-1]), key=operator.itemgetter(1))
        assert maxIndex>=0
        
        # now proceed backwards, following backpointers
        for i in range(nTokens)[::-1]:
            sent[i] = sent[i]._replace(prediction=labels[maxIndex])
            maxIndex = dpBackPointers[i,maxIndex]
            
        return maxScore


cdef i_viterbi(sent, o0Feats, float[:] weights, 
              float[:, :] dpValuesFwd, float[:, :] dpValuesBwd, int[:, :] dpBackPointers, 
              float[:, :] o0Scores, float[:,:,:] o1FeatWeights, labels, freqSortedLabelIndices, featureIndexes, includeLossTerm=False, costAugVal=0.0, useBIO=False):
        '''Uses the iterative Viterbi algorithm of Kaji et al. 2010 for staggered decoding (cf. Huang et al. 2012). 
        With proper caching and pruning this is much faster than standard Viterbi. 
        Updates the predicted labels in 'sent'. Used in both training and testing.
        (Assertions are commented out for speed.)'''
                
        indexerSize = len(featureIndexes)
        
        hasFOF = featureExtractor.hasFirstOrderFeatures()
        
        cdef int nTokens, nLabels, i, k, k2, kq, l, l2, lq, maxIndex, q, direc, last, backpointer
        cdef float score, score0, maxScore, INF, NEGINF, lower_bound
        cdef float[:,:] dpValues
        
        INF = float('inf')
        NEGINF = float('-inf')
        nTokens = len(sent)
        nLabels = len(labels)
        
        dpValuesFwd[:,:] = NEGINF
        dpValuesBwd[:,:] = NEGINF
        dpBackPointers[:,:] = -1
        o0Scores[:,:] = INF # INF means not yet computed
        # do not have to initialize o1FeatWeights each time because its contents do not depend on the sentence.
        
        prevLabel = None
        
        for i, tok in enumerate(sent):
            sent[i] = tok._replace(prediction=None)
        
        latticeColumnSize = [1]*len(sent)   # number of active labels for each token
        nExpansions = [0]*len(sent)
        
        pruned = [set() for t in range(nTokens)]
        
        lower_bound = NEGINF
        
        iterate = True
        nIters = 0
        firstiter = True
        direc = -1   # -1 for backward Viterbi, 1 for forward
        while iterate: # iterations
            iterate = False
            direc = -direc
            if direc==1:
                dpValues = dpValuesFwd
            else:
                dpValues = dpValuesBwd
            
            dpValuesActive = [[None]*latticeColumnSize[q] for q in range(nTokens)]
            
            for i in range(nTokens)[::direc]:
                columnLabels = freqSortedLabelIndices[:latticeColumnSize[i]+1]
                columnLabelsDegenerate = freqSortedLabelIndices[latticeColumnSize[i]:]
                
                o0FeatureMap = o0Feats[i]
                
                for l,lIsCollapsed in zip(columnLabels, [False]*latticeColumnSize[i]+[True]):
                    if (not lIsCollapsed) and l in pruned[i]: continue
                    
                    maxScore = NEGINF
                    maxIndex = -1
                    maxScoreActive = NEGINF
                    
                    # score for zero-order features
                    score0s = []
                    for l2 in ([l] if not lIsCollapsed else columnLabelsDegenerate):
                        if l2 in pruned[i]: continue
                        
                        if o0Scores[i,l2]==INF:    # compute and store the zero-order score for this label at this position
                            label = labels[l2]
                            if i==0 and not legalTagBigram(None, label, useBIO):
                                score0 = NEGINF
                            else:
                                score0 = _score(o0FeatureMap, weights, l2, indexerSize)
                                # cost-augmented decoding
                                if label!=sent[i].gold:
                                    if includeLossTerm:
                                        score0 += 1.0   # base cost of any error
                                    if label=='O' and sent[i].gold=='B':
                                        score0 += costAugVal    # recall-oriented penalty (for erroneously predicting 'O')
                                
                            o0Scores[i,l2] = score0
                        else:
                            score0 = o0Scores[i,l2]
                        
                        score0s.append(score0)
                    score0 = max(score0s) if score0s else NEGINF
                    
                    # consider each possible previous label
                    if (direc==1 and i==0) or (direc==-1 and i==nTokens-1): # beginning of the path
                        maxScore = score = score0
                        maxIndex = freqSortedLabelIndices[0]    # doesn't matter--start of path through lattice
                        '''assert maxIndex>=0'''
                        if not lIsCollapsed:
                            maxScoreActive = maxScore
                    else:   # look backwards
                        kcolumnLabels = freqSortedLabelIndices[:latticeColumnSize[i-direc]+1]
                        kcolumnLabelsDegenerate = freqSortedLabelIndices[latticeColumnSize[i-direc]:]
                        for k,kIsCollapsed in zip(kcolumnLabels, [False]*latticeColumnSize[i-direc]+[True]):
                            if (not kIsCollapsed) and k in pruned[i-direc]: continue
                            
                            # indices for the cached labels, which may be degenerate
                            lq = nLabels + nExpansions[i] if lIsCollapsed else l
                            kq = nLabels + nExpansions[i-direc] if kIsCollapsed else k
                            
                            score = o1FeatWeights[(direc+1)//2,lq,kq]
                            if score==INF:
                            
                                score1s = []
                                for k2 in ([k] if not kIsCollapsed else kcolumnLabelsDegenerate):
                                    if k2 in pruned[i-direc]: continue
                                    
                                    # score of moving from label k at the previous position to the current position (i) and label (l)
                                    if hasFOF or useBIO:
                                        # TODO: generalize this to allow other kinds of first-order features?
                                        # (may require resorting to bounds for efficiency)
                                        for l2 in ([l] if not lIsCollapsed else columnLabelsDegenerate):
                                            if l2 in pruned[i]: continue
                                            
                                            if o1FeatWeights[(direc+1)//2,l2,k2]==INF:    # don't bother to consult the matrix if only checking BIO constraint
                                                label = labels[l2]
                                                kLabel = labels[k2]
                                                if direc==1:
                                                    leftLabel, rightLabel = kLabel, label
                                                else:
                                                    leftLabel, rightLabel = label, kLabel
                                                
                                                if not legalTagBigram(leftLabel, rightLabel, useBIO):
                                                    o1FeatWeights[(direc+1)//2,l2,k2] = NEGINF
                                                elif hasFOF:
                                                    o1FeatWeights[(direc+1)//2,l2,k2] = weights[_ground0(featureIndexes[('prevLabel=',leftLabel)], (l2 if direc==1 else k2), indexerSize)]
                                                else:
                                                    o1FeatWeights[(direc+1)//2,l2,k2] = 0.0
                                                
                                            score1s.append(o1FeatWeights[(direc+1)//2,l2,k2])
                                    else:
                                        score1s.append(0.0)
                                
                                score = max(score1s) if score1s else NEGINF
                                o1FeatWeights[(direc+1)//2,lq,kq] = score
                                
                            
                            # compute correct score based on previous scores
                            score1 = dpValues[i-direc,k]    # NOT k2, because we are searching the degenerate lattice!
                            
                            score += score1
                            
                            # the score for the previous label is added on separately here,
                            # in order to avoid computing the whole score--which only 
                            # depends on the previous label for one feature--a quadratic 
                            # number of times
                            # TODO: plus vs. times doesn't matter here, right? use plus to avoid numeric overflow
                            
                            score += score0
                            
                            
                            # find the max of the combined score at the current position
                            # and store the backpointer accordingly
                            if score>maxScore:
                                maxScore = score
                                maxIndex = k
                            if not lIsCollapsed and not kIsCollapsed:
                                if score>NEGINF:    # NEGINF if BIO constraint is violated
                                    scoreActive = dpValuesActive[i-direc][freqSortedLabelIndices.index(k)] + score0
                                    if hasFOF:
                                        scoreActive += o1FeatWeights[(direc+1)//2,l,k]
                                    if scoreActive>maxScoreActive:
                                        maxScoreActive = scoreActive
                            
                    dpValues[i,l] = maxScore
                    dpBackPointers[i,l] = maxIndex
                    if not lIsCollapsed:
                        dpValuesActive[i][freqSortedLabelIndices.index(l)] = maxScoreActive
                        if not firstiter and dpValuesFwd[i,l]>NEGINF and dpValuesBwd[i,l]>NEGINF:
                            # pruning
                            # the >NEGINF checks are to ensure that the label wasn't newly activated 
                            # in this round, and therefore has both forward and backward values!
                            '''
                            assert score0==o0Scores[i,l]
                            assert maxScore in (dpValuesFwd[i,l],dpValuesBwd[i,l])
                            '''
                            upper_bound_this_node = dpValuesFwd[i,l]+dpValuesBwd[i,l]-score0
                            if upper_bound_this_node < lower_bound:
                                upper_bound = max(dpValues[nTokens-1 if direc==1 else 0])
                                pruned[i].add(l)
            
            
            # decode from the lattice
            # extract predictions from backpointers
            
            # first, find the best label for the last token
            last = nTokens-1 if direc==1 else 0
            backpointer, upper_bound = max([(q, dpValues[last][q]) for q in freqSortedLabelIndices[:latticeColumnSize[last]+1]], key=operator.itemgetter(1))
            best_active = max(dpValuesActive[last])
            
            
            if lower_bound < best_active:
                lower_bound = best_active
            
            
            # now proceed in the opposite direction, following backpointers
            reachedPrunedLabel = False
            for i in range(nTokens)[::-direc]:
                if latticeColumnSize[i]<nLabels and backpointer==freqSortedLabelIndices[latticeColumnSize[i]]:    # best decoding uses a collapsed label at this position
                    # column-wise expansion
                    latticeColumnSize[i] = min(latticeColumnSize[i]*2, nLabels)
                    nExpansions[i] += 1
                    iterate = True
                else:   # best decoding uses an active label at this position
                    sent[i] = sent[i]._replace(prediction=labels[backpointer])
                
                if backpointer in pruned[i]:
                    reachedPrunedLabel = True
                '''assert backpointer>=0,(backpointer,direc,i)'''
                backpointer = dpBackPointers[i,backpointer]
                
            
            '''
            if iterate:
                # calculate lower bound by decoding using only active labels
                assert lower_bound<=upper_bound,(lower_bound,upper_bound)
            else:
                assert lower_bound==upper_bound,(lower_bound,upper_bound)
            
            assert iterate or not reachedPrunedLabel,upper_bound
            '''
            firstiter = False
            nIters += 1
            
        '''assert upper_bound==best_active,(upper_bound,best_active)'''
        return upper_bound

class DiscriminativeTagger(object):
    def __init__(self, cutoff=None, defaultY=None):
        self._featureIndexes = features.SequentialStringIndexer(cutoff=cutoff)
        self._weights = None
        self._labels = []
        self._defaultY = defaultY
        self._labelC = Counter()
        #self._rgen = random.Random(1234567)
        
    @staticmethod
    def loadLabelList(labelFile, legacy0):
        '''
        Load a list of possible labels. This must be done before training 
        so that the feature vector has the appropriate dimensions.
        '''
        labels = []
        for ln in labelFile:
            if ln[:-1]:
                l = ln[:-1].decode('utf-8')
                if legacy0 and l=='0':
                    l = 'O'
                labels.append(l)
        return labels
    
    @staticmethod
    def removeExtraLabels(label, labels):
        '''
        Remove labels for adjectives and adverbs, which the SST does not address
        because they are lumped together in WordNet.
        '''
        #/*if(label.contains("-adj.") || label.contains("-adv.") || label.endsWith(".other")){
        #    return "0";
        #}*/
        return label if label in labels else 'O'
    
    @staticmethod
    def loadSuperSenseData(path, labels):
        '''
        Load the BIO tagged supersense data from Semcor, as provided in 
        the SuperSenseTagger release (SEM_07.BI).
        We also use their POS labels, which presumably were what their 
        paper used.
        One difference is that this method expects the data to be converted 
        into a 3-column format with an extra newline between each sentence 
        (as in CoNLL data), which can be created from the SST data with 
        a short perl script. 
        '''
        res = []
        with codecs.open(path, 'r', 'utf-8') as inF:
            sent = LabeledSentence()
            for ln in inF:
                if not ln.strip():
                    if len(sent)>0:
                        res.append(sent)
                        sent = LabeledSentence()
                    continue
                parts = ln[:-1].split('\t')
                if len(parts)>3:
                    if parts[3]!='':
                        sent.articleId = parts[3]
                    parts = parts[:3]
                token, pos, label = parts
                label = DiscriminativeTagger.removeExtraLabels(label, labels)
                stemS = morph.stem(token,pos)
                sent.addToken(token=token, stem=stemS, pos=pos, goldLabel=label)
                
            if len(sent)>0:
                res.append(sent)
        
        return res
    
    def printWeights(self, out, weights=None):
        if weights is None:
            weights = self._weights
        cdef int indexerSize, index, i, d
        cdef float baseline
        indexerSize = len(self._featureIndexes)
        if self._defaultY is not None:
            d = self._labels.index(self._defaultY)
        for index,fname in sorted(self._featureIndexes.items(), key=lambda x: x[1]):
            baseline = 0.0
            if self._defaultY is not None:
                baseline = weights[_ground0(index,d,indexerSize)]
            for i,label in enumerate(self._labels):
                value = weights[_ground0(index,i,indexerSize)]
                if value==0.0:
                    value = 0
                if self._defaultY is not None:
                    print(label.encode('utf-8'), fname, value, value-baseline, sep='\t', file=out)
                else:
                    print(label.encode('utf-8'), fname, value, sep='\t', file=out)
            print(file=out)
            
    def tagStandardInput(self):
        # TODO: this depends on MaxentTagger from the Stanford tools for decoding
        pass
    
    def getGroundedFeatureIndex(self, liftedFeatureIndex, labelIndex):
        return liftedFeatureIndex + labelIndex*len(self._featureIndexes)
    
    def _perceptronUpdate(self, sent, o0Feats, float[:] currentWeights, timestep, runningAverageWeights, learningRate=1.0):
        '''
        Update weights by iterating through the sequence, and at each token position 
        adding the feature vector for the correct label and subtracting the feature 
        vector for the predicted label.
        @param sent: the sentence, including gold and predicted tags
        @param o0Feats: active lifted zero-order features for each token
        @param currentWeights: latest value of the parameter value
        @param timestamp: number of previou updates that have been applied
        @param runningAverageWeights: average of the 'timestamp' previous weight vectors
        @return: number of weights updated
        '''
        
        if sent.predictionsAreCorrect(): return 0
        
        updates = set()
        
        cdef int featIndex
        
        for i,(tkn,o0FeatureMap) in enumerate(zip(sent, o0Feats)):
            pred = self._labels.index(tkn.prediction)
            gold = self._labels.index(tkn.gold)
            
            if pred==gold: continue # TODO: is this correct if we are being cost-augmented?
            
            # update gold label feature weights
            
            # zero-order features
            #o0FeatureMap = featureExtractor.extractFeatureValues(sent, i, usePredictedLabels=False, orders={0}, indexer=self._featureIndexes)
            for h,v in o0FeatureMap.items():
                featIndex = _ground(h, gold, self._featureIndexes)
                currentWeights[featIndex] += learningRate * v
                updates.add(featIndex)
                
            # first-order features
            if featureExtractor.hasFirstOrderFeatures() and i>0:
                o1FeatureMap = featureExtractor.extractFeatureValues(sent, i, usePredictedLabels=False, orders={1}, indexer=self._featureIndexes)
                for h,v in o1FeatureMap.items():
                    featIndex = _ground(h, gold, self._featureIndexes)
                    currentWeights[featIndex] += learningRate * v
                    updates.add(featIndex)
            
            if not o0FeatureMap and not o1FeatureMap:
                raise Exception('No features found for this token')
        
            
            # update predicted label feature weights
            
            # zero-order features
            #o0FeatureMap = featureExtractor.extractFeatureValues(sent, i, usePredictedLabels=True, orders={0}, indexer=self._featureIndexes)
            for h,v in o0FeatureMap.items():
                featIndex = _ground(h, pred, self._featureIndexes)
                currentWeights[featIndex] -= learningRate * v
                updates.add(featIndex)
                
            # first-order features
            if featureExtractor.hasFirstOrderFeatures() and i>0:
                o1FeatureMap = featureExtractor.extractFeatureValues(sent, i, usePredictedLabels=True, orders={1}, indexer=self._featureIndexes)
                for h,v in o1FeatureMap.items():
                    featIndex = _ground(h, pred, self._featureIndexes)
                    currentWeights[featIndex] -= learningRate * v
                    updates.add(featIndex)
            
            if not o0FeatureMap and not o1FeatureMap:
                raise Exception('No features found for this token')
            
        for featIndex,current in enumerate(currentWeights): # need to update averages for *all* weights, or else store timestamps and use lazy updating, making sure to update all weights after the last iteration (http://blog.smola.org/post/943941371/lazy-updates-for-generic-regularization-in-sgd)
            runningAverageWeights[featIndex] = (timestep*runningAverageWeights[featIndex] + currentWeights[featIndex])/(timestep+1)
            
        return len(updates)
    
    def _createFeatures(self, trainingData, sentIndices=slice(0,None)):
        '''Before training, loop through the training data once 
        to instantiate all possible features, and create the weight 
        vector'''
        
        print('instantiating features', file=sys.stderr)
        
        # instantiate first-order features for all possible previous labels
        o1Feats = set() if featureExtractor.hasFirstOrderFeatures() else None
        
        # create a feature for each label as the previous label
        # TODO: if using a caching format, consider doing this even if not using first-order features
        if featureExtractor.hasFirstOrderFeatures():
            _o1Feats = [0]*len(self._labels)
            for l,lbl in enumerate(self._labels):
                key = ('prevLabel=',lbl)  # TODO: assumes this is the only first-order feature
                self._featureIndexes.add(key)
                
        # instantiate the rest of the features
        ORDERS0 = {0}
        for nSent,sentAndFeats in enumerate(trainingData):
            if nSent<sentIndices.start: continue
            if sentIndices.stop is not None and nSent>sentIndices.stop: break
            
            # SupersenseFeaturizer will index new zero-order features as they are encountered
            """
            for i in range(len(sent)):
                # will index new features as they are encountered
                featureExtractor.extractFeatureValues(sent, i, usePredictedLabels=False, orders=ORDERS0, indexer=self._featureIndexes)
                '''
                for h,v in featureExtractor.extractFeatureValues(sent, i, usePredictedLabels=False, orders=ORDERS0, indexer=self._featureIndexes).items():
                    # TODO: first-order features handled above, so zero-order only here
                    self._featureIndexes.add(h)
                '''
            """
            
            # count labels
            self._labelC.update([tkn.gold for tkn in sentAndFeats[0]])
            
            if nSent%1000==0:
                print('.', file=sys.stderr, end='')
            elif nSent%100==0:
                print(',', file=sys.stderr, end='')
        
        trainingData.reset()
        
        # ensure these aren't filtered by a feature cutoff
        if self._featureIndexes._cutoff is not None:
            for label in self._labels:
                self._featureIndexes.setcount(('prevLabel=', label), self._featureIndexes._cutoff)
        
        self._featureIndexes.freeze()
        
        # now create the array of feature weights
        nWeights = len(self._labels)*len(self._featureIndexes)
        
        print(' done with',nSent,'sentences:',len(self._labels),'labels,',len(self._featureIndexes),'lifted features, size',nWeights,'weight vector', file=sys.stderr)
        print('label counts:',self._labelC, file=sys.stderr)
        
        self._freqSortedLabelIndices = list(range(len(self._labels)))
        self._freqSortedLabelIndices.sort(key=lambda l: self._labelC[l], reverse=True)
        
    
    def _computeScore(self, featureMap, weights, labelIndex):
        '''Compute the dot product of a set of feature values and the corresponding weights.'''
        if labelIndex==-1:
            return 0.0
        
        dotProduct = 0.0
        for h,v in featureMap.items():
            dotProduct += weights[self.getGroundedFeatureIndex(h, labelIndex)]*v
        return dotProduct
    
    def _viterbi(self, sent, o0Feats, float[:] weights, float[:, :] dpValuesFwd, float[:, :] dpValuesBwd, 
                 int[:, :] dpBackPointers, float[:, :] o0Scores, float[:, :, :] o1FeatWeights, 
                 includeLossTerm=False, costAugVal=0.0, useBIO=False):
        
        nTokens = len(sent)
        
        # expand the size of dynamic programming tables if necessary
        if len(dpValuesFwd)<nTokens:
            #dpValues = [[0.0]*len(self._labels) for t in range(int(nTokens*1.5))]
            #dpBackPointers = [[0]*len(self._labels) for t in range(int(nTokens*1.5))]
            dpValuesFwd = cvarray(shape=(int(nTokens*1.5), len(self._labels)), itemsize=sizeof(float), format='f')
            dpValuesBwd = cvarray(shape=(int(nTokens*1.5), len(self._labels)), itemsize=sizeof(float), format='f')
            dpBackPointers = cvarray(shape=(int(nTokens*1.5), len(self._labels)), itemsize=sizeof(int), format='i')
            o0Scores = cvarray(shape=(int(nTokens*1.5), len(self._labels)), itemsize=sizeof(float), format='f')
        
        METHOD = 'c'   # conventional and/or iterative Viterbi
        if 'c' in METHOD:
            c_score = c_viterbi(sent, o0Feats, weights, dpValuesFwd, dpBackPointers, self._labels, self._featureIndexes, includeLossTerm, costAugVal, useBIO)
            c_preds = [x.prediction for x in sent]
        if 'i' in METHOD:
            i_score = i_viterbi(sent, o0Feats, weights, dpValuesFwd, dpValuesBwd, dpBackPointers, o0Scores, o1FeatWeights, self._labels, self._freqSortedLabelIndices, self._featureIndexes, includeLossTerm, costAugVal, useBIO)
            i_preds = [x.prediction for x in sent]
        if 'c' in METHOD and 'i' in METHOD: # check that results match
            print(c_score,c_preds)
            print(i_score,i_preds)
            print('---')
            assert c_score==i_score,(c_score,i_score)

    def train(self, trainingData, savePrefix, instanceIndices=None, averaging=False, tuningData=None, earlyStopInterval=None, maxIters=2, developmentMode=False, useBIO=False, includeLossTerm=False, costAugVal=0.0, gamma=1.0):
        '''Train using the perceptron. See Collins paper on discriminative HMMs.'''
        assert maxIters>0,maxIters
        assert earlyStopInterval is None or tuningData is not None
        print('training with the perceptron for up to',maxIters,'iterations', 
              ('with early stopping by checking the tuning data every {} iterations'.format(abs(earlyStopInterval)) if earlyStopInterval is not None else ''),
              file=sys.stderr)
        
        # create feature vocabulary for the training data
        assert trainingData
        self._createFeatures(trainingData, sentIndices=instanceIndices)
        trainingData.enable_caching()   # don't cache before the featureset is finalized!
        
        # save features
        if developmentMode and savePrefix is not None:
            # print features before training
            with open(savePrefix+'.features', 'w') as outF:
                self.printFeatures(outF)
        
        prevNCorrect = prevTotCost = None
        nTuning = None
        
        prevWeights = None
        
        # training iterations: calls decode()
        for i,weights in enumerate(self.decode(trainingData, maxTrainIters=maxIters, averaging=averaging, 
                                               useBIO=useBIO, includeLossTerm=includeLossTerm, costAugVal=costAugVal)):
            
            # store the new weights in an attribute
            self._weights = weights
            print('l2(prevWeights) = {:.4}, l2(weights) = {:.4}'.format(l2norm(prevWeights or [0.0]),l2norm(weights)), file=sys.stderr)
            
            # if dev mode, save each model and human-readable weights file
            if developmentMode:
                #self.test()
                if savePrefix is not None:
                    self.saveModel(savePrefix+'.'+str(i))
                    with open(savePrefix+'.'+str(i)+'.weights', 'w') as outF:
                        self.printWeights(outF, weights)
                        
            
            if earlyStopInterval is not None and i<maxIters-1 and (i+1)%abs(earlyStopInterval)==0:
                # decode on tuning data and decide whether to stop
                next(self.decode(tuningData, maxTrainIters=0, averaging=averaging,
                      useBIO=useBIO, includeLossTerm=False, costAugVal=0.0))
                totCost = nCorrect = nTuning = 0
                for sent,o0Feats in tuningData:
                    # TODO: evaluate cost rather than tag accuracy?
                    nCorrect += sum(1 for tok in sent if tok.gold==tok.prediction)
                    totCost += sum(1+(costAugVal if tok.gold=='O' else 0) for tok in sent if tok.gold!=tok.prediction)
                    nTuning += len(sent)
                if prevNCorrect is not None:
                    if earlyStopInterval>0 and nCorrect <= prevNCorrect: # use accuracy as criterion
                        print('stopping early after iteration',i,
                              '. new tuning set acc {}/{}={:.2%}, previously {}/{}={:.2%}'.format(nCorrect,nTuning,nCorrect/nTuning,
                                                                                                  prevNCorrect,nTuning,prevNCorrect/nTuning),
                              file=sys.stderr)
                        self._weights = prevWeights # the last model that posted an improvement
                        break
                    elif earlyStopInterval<0 and totCost >= prevTotCost:   # use cost as criterion
                        print('stopping early after iteration',i,
                              '. new tuning set avg cost {}/{}={:.2%}, previously {}/{}={:.2%}'.format(totCost,nTuning,totCost/nTuning,
                                                                                                       prevTotCost,nTuning,prevTotCost/nTuning),
                              file=sys.stderr)
                        self._weights = prevWeights # the last model that posted an improvement
                        break
                prevNCorrect = nCorrect
                prevTotCost = totCost
                
            # hold on to the previous weights
            prevWeights = list(self._weights)
        
        # save model
        if savePrefix is not None:
            self.saveModel(savePrefix)
        
    
    def decode(self, data, maxTrainIters=0, averaging=False, useBIO=False, includeLossTerm=False, costAugVal=0.0, gamma=1.0):
        '''Decode a dataset under a model. Predictions are stored in the sentence within the call to _viterbi(). 
        If maxTrainIters is positive, update the weights. 
        After each iteration, the weights are yielded.'''
        
        print('decoding data type:', type(data), file=sys.stderr)
        print('learning:',bool(maxTrainIters),'averaging:',averaging,'BIO:',useBIO,'costAug:',includeLossTerm,costAugVal, file=sys.stderr)
        
        MAX_NUM_TOKENS = 200
        nLabels = len(self._labels) # number of (actual) labels
        nDegenerateLabels = int(math.ceil(math.log(nLabels,2)))  # for iterative Viterbi
        nWeights = len(self._labels)*len(self._featureIndexes)
        
        # create DP tables
        #dpValues = [[0.0]*nLabels for t in range(MAX_NUM_TOKENS)];
        #dpBackPointers = [[0]*nLabels for t in range(MAX_NUM_TOKENS)]
        
        dpValuesFwd = cvarray(shape=(MAX_NUM_TOKENS, nLabels), itemsize=sizeof(float), format='f')
        dpValuesBwd = cvarray(shape=(MAX_NUM_TOKENS, nLabels), itemsize=sizeof(float), format='f')
        dpBackPointers = cvarray(shape=(MAX_NUM_TOKENS, nLabels), itemsize=sizeof(int), format='i')
        o0Scores = cvarray(shape=(MAX_NUM_TOKENS, nLabels), itemsize=sizeof(float), format='f')
        o1FeatWeights = cvarray(shape=(2, nLabels+nDegenerateLabels, nLabels+nDegenerateLabels), itemsize=sizeof(float), format='f')
        o1FeatWeights[:,:,:] = float('inf')    # INF = uninitialized. note that the contents do not depend on the sentence.
        
        update = (maxTrainIters>0)   # training?
        
        #finalWeights = [0.0]*nWeights
        #currentWeights = [0.0]*nWeights
        
        # the model to decode with. if learning with averaging, contains the latest non-averaged weight vector. 
        # otherwise same as finalWeights.
        currentWeights = cvarray(shape=(nWeights,), itemsize=sizeof(float), format='f')
        # the model to be written to disk. will be yielded after each iteration. includes averaging if applicable.
        finalWeights = cvarray(shape=(nWeights,), itemsize=sizeof(float), format='f')
        # currentWeights & finalWeights are initialized to self._weights if set, otherwise 0
        
        '''
        if update:
            yield finalWeights  # TODO: debugging: don't train at all!
            return
        '''
        
        if self._weights is not None:  # initialize weights
            #assert len(self._weights)==nWeights    # can't len() an array?
            for i,w in enumerate(self._weights):
                finalWeights[i] = currentWeights[i] = w
        
        
        # tabulate accuracy at every 500 iterations
        nWordsProcessed = 0
        nWordsIncorrect = 0
        totalWordsProcessed = 0
        totalWordsIncorrect = 0
        totalInstancesProcessed = 0
        
        gamma_current = 1.0   # current learning rate
        
        for numIters in range(max(1,maxTrainIters)):
            if update:
                print('iter = ',numIters, file=sys.stderr)
            
            # TODO: shuffle the training data if not reading it incrementally?
            
            nWeightUpdates = 0
            
            for isent,(sent,o0Feats) in enumerate(data): # to limit the number of instances, see _createFeatures()
                
                self._viterbi(sent, o0Feats, currentWeights, dpValuesFwd, dpValuesBwd, dpBackPointers,
                              o0Scores, o1FeatWeights, includeLossTerm=includeLossTerm, costAugVal=costAugVal, 
                              useBIO=useBIO)
        
                if update:
                    gamma_current *= gamma
                    nWeightUpdates += self._perceptronUpdate(sent, o0Feats, currentWeights, totalInstancesProcessed, finalWeights, learningRate=gamma_current)
                    # will update currentWeights as well as running average in finalWeights
                    o1FeatWeights[:,:,:] = float('inf') # clear the bigram feature weights cache (they will be recomputed the next time we decode)
                
                for i in range(len(sent)):
                    if sent[i].gold != sent[i].prediction:
                        nWordsIncorrect += 1
                        totalWordsIncorrect += 1
                nWordsProcessed += len(sent)
                totalWordsProcessed += len(sent)
                totalInstancesProcessed += 1
                #print(',', end='', file=sys.stderr)
                
                if isent==0:    # print the tagging of the first sentence in the dataset
                    print(' '.join(tkn.prediction for tkn in sent).encode('utf-8'), file=sys.stderr)
                
                if totalInstancesProcessed%100==0:
                    print('totalInstancesProcessed = ',totalInstancesProcessed, file=sys.stderr)
                    print('pct. correct words in last 100 inst.: {:.2%}'.format((nWordsProcessed-nWordsIncorrect)/nWordsProcessed), file=sys.stderr)
                    nWordsIncorrect = nWordsProcessed = 0
                elif totalInstancesProcessed%10==0:
                    print('.', file=sys.stderr, end='')
            
            
            if update and not averaging:
                finalWeights = currentWeights
            
            print('l2(currentWeights) = {:.4}, l2(finalWeights) = {:.4}'.format(l2norm(currentWeights), l2norm(finalWeights)), file=sys.stderr)
            print('word accuracy over {} words in {} instances: {:.2%}'.format(totalWordsProcessed, totalInstancesProcessed, (totalWordsProcessed-totalWordsIncorrect)/totalWordsProcessed), file=sys.stderr)
                
            yield finalWeights
            
            if update:
                print('weight updates this iteration:',nWeightUpdates, file=sys.stderr)
                if nWeightUpdates==0:
                    print('converged! stopped training', file=sys.stderr)
                    break

    def printFeatures(self, out):
        print(len(self._featureIndexes),'lifted features x',len(self._labels),'labels =',len(self._featureIndexes)*len(self._labels),'grounded features', file=out)
        print('labels:',self._labels,'\n', file=out)
        for fname in sorted(self._featureIndexes.strings):
            print(''.join(fname).encode('utf-8'), file=out)
    
    def saveModel(self, savePrefix):
        import cPickle
        saveFP = savePrefix+'.pickle'
        # lists but not arrays can be pickled. so temporarily store a list.
        weights = self._weights
        if not isinstance(self._weights, list):
            self._weights = list(weights)
        with open(saveFP, 'wb') as saveF:
            cPickle.dump(self, saveF)
        self._weights = weights
    
    @staticmethod
    def loadModel(savePrefix):
        import cPickle
        saveFP = savePrefix+'.pickle'
        with open(saveFP, 'rb') as saveF:
            model = cPickle.load(saveF)
        return model
        
    def test(self, weights):
        raise NotImplemented()
        '''
        if self._testData is None: return
        
        for sent,o0Feats in self._testData:
            self._viterbi(sent, o0Feats, weights, dpValues, dpBackPointers, includeLossTerm, costAugVal, useBIO)
        
        self.evaluatePredictions(self._testData, self._labels);
        '''
    
def main():
    import argparse
    
    opts = argparse.ArgumentParser(description='Learn or predict from a discriminative tagging model')
    
    def flag(name, description, ftype=str, **kwargs):
        opts.add_argument(('--' if len(name)>1 else '-')+name, type=ftype, help=description, **kwargs)
    def inflag(name, description, ftype=argparse.FileType('r'), **kwargs):
        flag(name, description, ftype=ftype, **kwargs)
    def outflag(name, description, ftype=argparse.FileType('w'), **kwargs):
        flag(name, description, ftype=ftype, **kwargs)
    def boolflag(name, description, default=False, **kwargs):
        opts.add_argument(('--' if len(name)>1 else '-')+name, action='store_false' if default else 'store_true', help=description, **kwargs)
    
    flag("train", "Path to training data feature file") #inflag
    flag("max-train-instances", "During training, truncate the data to the specified number of instances", ftype=int)
    boolflag("disk", "Load instances from the feature file in each pass through the training data, rather than keeping the full training data in memory")
    flag("iters", "Number of passes through the training data", ftype=int, default=1)
    flag("early-stop", "Interval (number of iterations) between checks on the test data to decide whether to stop early. If negative, cost rather than tagging accuracy is used as the stopping criterion.", ftype=int, default=None)
    inflag("test", "Path to test data for a CoNLL-style evaluation; scores will be printed to stderr (following training, if applicable)")
    #flag("max-test-instances", "During testing, truncate the data to the specified number of instances", ftype=int)
    boolflag("debug", "Whether to save the list of feature names (.features file) prior to training, as well as an intermediate model (serialized model file and text file with feature weights) after each iteration of training")
    inflag("YY", "List of possible labels, one label per line") # formerly: labels
    flag("defaultY", "Default (background) label: model weights represent deviation from this label")
    flag("save", "Save path for serialized model file (training only). Associated output files (with --debug) will add a suffix to this path.")
    flag("load", "Path to serialized model file (decoding only)")   #inflag
    #inflag("properties", "Properties file with option defaults", default="tagger.properties")
    #boolflag("mira"),
    boolflag("weights", "Write feature weights to stdout after training")
    flag("test-predict", "Path to feature file on which to make predictions (following training, if applicable); predictions will be written to stdout. (Will be ignored if --test is supplied.)")
    #inflag
    
    # formerly only allowed in properties file
    flag("bio", "Constrain label bigrams in decoding such that the 'O' label is never followed by a label beginning with 'I'", nargs='?', const=True, default=False, choices={'NO_SINGLETON_B'})
    boolflag("legacy0", "BIO scheme uses '0' instead of 'O'")
    boolflag("includeLossTerm", "Incur a cost of (at least) 1 whenever making a tagging error during training.")
    flag("costAug", "Value of cost penalty for errors against recall (for recall-oriented learning)", ftype=float, default=0.0)
    boolflag("excludeFirstOrder", "Do not include label bigram features", default=False)
    
    # formerly: "useFeatureNumber"
    flag("excludeFeatures","Comma-separated list of (0-based) column numbers to ignore when reading feature files. (Do not specify column 0; use --no-lex instead.)", default='')
    flag("cutoff", "Threshold (minimum number of occurrences) of a feature for it to be included in the model", ftype=int, default=None)
    flag("gamma","Base of learning rate (exponent is number of instances seen so far)", ftype=float, default=1.0)
    boolflag("no-lex", "Don't include features for current and context token strings")
    boolflag("no-averaging", "Don't use averaging in perceptron training")
    
    # features
    boolflag("mwe", "Multiword expressions featureset")
    boolflag("bigrams", "Token bigram features")
    boolflag("cxt-pos-filter", "Filter bigram features based on the POS pairs")
    boolflag("clusters", "Word cluster features")
    flag("cluster-file", "Path to file with word clusters", default=supersenseFeatureExtractor._options['clusterFile'])
    boolflag("pos-neighbors", "POS neighbor features")
    
    inflag("lex", "Lexicons to load for lookup features", nargs='*')
    
    args = opts.parse_args()
    
    if args.train is None and args.load is None:
        raise Exception('Missing argument: --train or --load')
    if args.YY is None and args.load is None:
        raise Exception('Missing argument: --YY')
    
    global featureExtractor
    if args.mwe:
        import mweFeatures as featureExtractor
    else:
        import supersenseFeatureExtractor as featureExtractor
    
    featureExtractor.registerOpts(args)
    
    testData = None
    
    if args.load is not None:
        print('loading model from',args.load,'...', file=sys.stderr)
        t = DiscriminativeTagger.loadModel(args.load)
        # override options used during training that may be different for prediction
        #t.setBinaryFeats(False)
        print('done.', file=sys.stderr)
    else:
        t = DiscriminativeTagger(cutoff=args.cutoff, defaultY=args.defaultY)
        #t.setBinaryFeats(False)
        labels = DiscriminativeTagger.loadLabelList(args.YY, args.legacy0)
        if t._defaultY is not None:
            assert t._defaultY in labels,'Default label missing from list of all labels: {}'.format(t._defaultY)
        t._labels = labels  # TODO: "private" access
        #t._labels = ['0', 'B-noun.person', 'I-noun.person']  # TODO: debugging purposes
        
        print('training model from',args.train,'...', file=sys.stderr)
        
        if not args.disk:
            #data = DiscriminativeTagger.loadSuperSenseData(args.train, labels)
            trainingData = SupersenseFeaturizer(featureExtractor, SupersenseDataSet(args.train, t._labels, legacy0=args.legacy0), t._featureIndexes, cache_features=False)
            if args.test is not None or args.test_predict is not None:
                testData = SupersenseFeaturizer(featureExtractor, SupersenseDataSet(args.test or args.test_predict, t._labels, legacy0=args.legacy0), t._featureIndexes, cache_features=False)
                
            t.train(trainingData, args.save, maxIters=args.iters, instanceIndices=slice(0,args.max_train_instances), averaging=(not args.no_averaging), 
                    earlyStopInterval=args.early_stop if (args.test or args.test_predict) else None, 
                    tuningData=testData,
                    developmentMode=args.debug, 
                    useBIO=args.bio, includeLossTerm=args.includeLossTerm, costAugVal=args.costAug, gamma=args.gamma)
            
            del trainingData
        else:
            raise NotImplemented()
    
    
    if args.test is not None or args.test_predict is not None:
        if testData is None:
            testData = SupersenseFeaturizer(featureExtractor, SupersenseDataSet(args.test or args.test_predict, t._labels, legacy0=args.legacy0), t._featureIndexes, cache_features=False)
        
        next(t.decode(testData, maxTrainIters=0, averaging=(not args.no_averaging),
                      useBIO=args.bio, includeLossTerm=False, costAugVal=0.0))
        
        if not args.test:
            # print predictions
            for sent,o0Feats in testData:
                for tok in sent:
                    print(tok.token.encode('utf-8'), tok.prediction.encode('utf-8'), sep='\t')
                print()
    
    elif args.weights:
        t.printWeights(sys.stdout)
    else:
        t.tagStandardInput()

if __name__=='__main__':
    #import cProfile
    #cProfile.run('main()')
    try:
        main()
    except KeyboardInterrupt:
        raise
