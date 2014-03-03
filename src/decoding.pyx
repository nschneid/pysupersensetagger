'''
Viterbi, scoring implementations. Called from DiscriminativeTagger.decode().
'''
from __future__ import print_function, division
import operator

#cimport cython

from pyutil.memoize import memoize

cdef float _score(object featureMap, float[:] weights, int labelIndex, int indexerSize):
        '''Compute the dot product of a set of feature values and the corresponding weights.'''
        if labelIndex==-1:
            return 0.0
        
        dotProduct = 0.0
        for h,v in featureMap.items():
            dotProduct += weights[_ground0(h, labelIndex, indexerSize)]*v
        return dotProduct



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

cdef c_viterbi(sent, o0Feats, featureExtractor, float[:] weights, 
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
                    if (label=='O' or label=='o') and (sent[i].gold=='B' or sent[i].gold=='b'):
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
                        
                        # the score for the previous label is added on separately here,
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


cdef i_viterbi(sent, o0Feats, featureExtractor, float[:] weights, 
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
                                    if (label=='O' or label=='o') and (sent[i].gold=='B' or sent[i].gold=='b'):
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
