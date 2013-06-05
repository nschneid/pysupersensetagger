# cython: profile=True
# cython: infer_types=True
'''
Created on Jul 24, 2012

@author: Nathan Schneider (nschneid)
'''
from __future__ import print_function, division
import sys, codecs, random, operator
from collections import defaultdict, Counter

cimport cython
from cython.view cimport array as cvarray

from labeledSentence import LabeledSentence
import supersenseFeatureExtractor, morph

from dataFeaturizer import SupersenseDataSet, SupersenseFeaturizer

def memoize(f):
    """
    Memoization decorator for a function taking one or more arguments.
    Source: http://code.activestate.com/recipes/578231-probably-the-fastest-memoization-decorator-in-the-/#c4 
    """
    class memodict(dict):
        def __getitem__(self, *key):
            return dict.__getitem__(self, key)

        def __missing__(self, key):
            ret = self[key] = f(*key)
            return ret

    return memodict().__getitem__


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
        if useBIO and lbl2[0]=='I':
            if lbl1 is None or lbl1=='O':
                return False    # disallow O followed by an I tag
            if (len(lbl1)>1)!=(len(lbl2)>1):
                return False    # only allow I without class if previous tag has no class
            if len(lbl2)>1 and lbl1[2:]!=lbl2[2:]:
                return False    # disallow an I tag following a tag with a different class
        return True

cdef c_viterbi(sent, o0Feats, float[:] weights, 
              float[:, :] dpValues, int[:, :] dpBackPointers, 
              labels, featureIndexes, o1FeatWeights, includeLossTerm=False, costAugVal=0.0, useBIO=False):
        '''Uses the Viterbi algorithm to decode, i.e. find the best labels for the sequence 
        under the current weight vector. Updates the predicted labels in 'sent'. 
        Used in both training and testing.'''
        
        indexerSize = len(featureIndexes)
        
        
        hasFOF = supersenseFeatureExtractor.hasFirstOrderFeatures()
        
        cdef int nTokens, i, k, l, maxIndex
        cdef float score, score0, maxScore, NEGINF
        
        NEGINF = float('-inf')
        nTokens = len(sent)
        
            
        prevLabel = None
        
        #cdef int i, l, k, maxIndex
        #cdef float score, score0, maxScore
        
        for i, tok in enumerate(sent):
            sent[i] = tok._replace(prediction=None)
        
        for i in range(nTokens):
            #o0FeatureMap = supersenseFeatureExtractor.extractFeatureValues(sent, i, usePredictedLabels=True, orders={0}, indexer=self._featureIndexes)
            o0FeatureMap = o0Feats[i]
            
            for l,label in enumerate(labels):
                maxScore = NEGINF
                maxIndex = -1
                
                # score for zero-order features
                score0 = _score(o0FeatureMap, weights, l, indexerSize)
                
                # cost-augmented decoding
                if label!=sent[i].gold:
                    if includeLossTerm:
                        score0 += 1.0   # base cost of any error
                    if label=='O':
                        score0 += costAugVal    # recall-oriented penalty (for erroneously predicting 'O')
                
                # consider each possible previous label
                for k,prevLabel in enumerate(labels):
                    if not legalTagBigram(None if i==0 else prevLabel, label, useBIO):
                        continue
                    
                    # compute correct score based on previous scores
                    score = 0.0
                    if i>0:
                        #sent[i-1] = sent[i-1]._replace(prediction=prevLabel)
                        score = dpValues[i-1,k]
                    
                    # the score for the previou label is added on separately here,
                    # in order to avoid computing the whole score--which only 
                    # depends on the previous label for one feature--a quadratic 
                    # number of times
                    # TODO: plus vs. times doesn't matter here, right? use plus to avoid numeric overflow
                    
                    # score of moving from label k at the previous position to the current position (i) and label (l)
                    score += score0
                    if hasFOF and i>0:
                        '''
                        o1FeatureMap = supersenseFeatureExtractor.extractFeatureValues(sent, i, usePredictedLabels=True, orders={1}, indexer=self._featureIndexes)
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
                    
                    # if this is the first token, there is only one possible 
                    # previous label
                    if i==0:
                        break
                    
                dpValues[i,l] = maxScore
                dpBackPointers[i,l] = maxIndex
        
        # decode from the lattice
        # extract predictions from backpointers
        
        # first, find the best label for the last token
        maxIndex, maxScore = max(enumerate(dpValues[nTokens-1]), key=operator.itemgetter(1))
        
        # now proceed backwards, following backpointers
        for i in range(nTokens-1,-1,-1):
            sent[i] = sent[i]._replace(prediction=labels[maxIndex])
            maxIndex = dpBackPointers[i,maxIndex]
        

class DiscriminativeTagger(object):
    def __init__(self):
        self._featureIndexes = supersenseFeatureExtractor.SequentialStringIndexer()
        self._weights = None
        self._labels = []
        self._labelC = Counter()
        #self._rgen = random.Random(1234567)
        
    @staticmethod
    def loadLabelList(labelFile):
        '''
        Load a list of possible labels. This must be done before training 
        so that the feature vector has the appropriate dimensions.
        '''
        labels = []
        with codecs.open(labelFile, 'r', 'utf-8') as labelF:
            for ln in labelF:
                if ln[:-1]:
                    labels.append(ln[:-1])
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
        return label if label in labels else '0'
    
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
        cdef int indexerSize, index, i
        indexerSize = len(self._featureIndexes)
        for index,fname in sorted(self._featureIndexes.items(), key=lambda x: x[1]):
            for i,label in enumerate(self._labels):
                value = weights[_ground0(index,i,indexerSize)]
                if value!=0.0:
                    print(label, fname, value, sep='\t', file=out)
            print(file=out)
            
    def tagStandardInput(self):
        # TODO: this depends on MaxentTagger from the Stanford tools for decoding
        pass
    
    def getGroundedFeatureIndex(self, liftedFeatureIndex, labelIndex):
        return liftedFeatureIndex + labelIndex*len(self._featureIndexes)
    
    def _perceptronUpdate(self, sent, o0Feats, float[:] currentWeights, timestep, runningAverageWeights):
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
            #o0FeatureMap = supersenseFeatureExtractor.extractFeatureValues(sent, i, usePredictedLabels=False, orders={0}, indexer=self._featureIndexes)
            for h,v in o0FeatureMap.items():
                featIndex = _ground(h, gold, self._featureIndexes)
                currentWeights[featIndex] += v
                updates.add(featIndex)
                
            # first-order features
            if supersenseFeatureExtractor.hasFirstOrderFeatures() and i>0:
                o1FeatureMap = supersenseFeatureExtractor.extractFeatureValues(sent, i, usePredictedLabels=False, orders={1}, indexer=self._featureIndexes)
                for h,v in o1FeatureMap.items():
                    featIndex = _ground(h, gold, self._featureIndexes)
                    currentWeights[featIndex] += v
                    updates.add(featIndex)
            
            if not o0FeatureMap and not o1FeatureMap:
                raise Exception('No features found for this token')
        
            
            # update predicted label feature weights
            
            # zero-order features
            #o0FeatureMap = supersenseFeatureExtractor.extractFeatureValues(sent, i, usePredictedLabels=True, orders={0}, indexer=self._featureIndexes)
            for h,v in o0FeatureMap.items():
                featIndex = _ground(h, pred, self._featureIndexes)
                currentWeights[featIndex] -= v
                updates.add(featIndex)
                
            # first-order features
            if supersenseFeatureExtractor.hasFirstOrderFeatures() and i>0:
                o1FeatureMap = supersenseFeatureExtractor.extractFeatureValues(sent, i, usePredictedLabels=True, orders={1}, indexer=self._featureIndexes)
                for h,v in o1FeatureMap.items():
                    featIndex = _ground(h, pred, self._featureIndexes)
                    currentWeights[featIndex] -= v
                    updates.add(featIndex)
            
            if not o0FeatureMap and not o1FeatureMap:
                raise Exception('No features found for this token')
            
        for featIndex in updates:
            runningAverageWeights[featIndex] = (timestep*runningAverageWeights[featIndex] + currentWeights[featIndex])/(timestep+1)
            
        return len(updates)
    
    def _createFeatures(self, trainingData, sentIndices=slice(0,100)):
        '''Before training, loop through the training data once 
        to instantiate all possible features, and create the weight 
        vector'''
        
        print('instantiating features', file=sys.stderr)
        
        # instantiate first-order features for all possible previous labels
        o1Feats = set() if supersenseFeatureExtractor.hasFirstOrderFeatures() else None
        
        # create a feature for each label as the previous label
        # TODO: if using a caching format, consider doing this even if not using first-order features
        if supersenseFeatureExtractor.hasFirstOrderFeatures():
            _o1Feats = [0]*len(self._labels)
            for l,lbl in enumerate(self._labels):
                key = ('prevLabel=',lbl)  # TODO: assumes this is the only first-order feature
                self._featureIndexes.add(key)
                
        # instantiate the rest of the features
        ORDERS0 = {0}
        for nSent,sentAndFeats in enumerate(trainingData):
            if nSent<sentIndices.start: continue
            if nSent>sentIndices.stop: break
            
            # SupersenseFeaturizer will index new zero-order features as they are encountered
            """
            for i in range(len(sent)):
                # will index new features as they are encountered
                supersenseFeatureExtractor.extractFeatureValues(sent, i, usePredictedLabels=False, orders=ORDERS0, indexer=self._featureIndexes)
                '''
                for h,v in supersenseFeatureExtractor.extractFeatureValues(sent, i, usePredictedLabels=False, orders=ORDERS0, indexer=self._featureIndexes).items():
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
        
        
        # now create the array of feature weights
        nWeights = len(self._labels)*len(self._featureIndexes)
        
        print(' done with',nSent,'sentences:',len(self._labels),'labels,',len(self._featureIndexes),'lifted features, size',nWeights,'weight vector', file=sys.stderr)
        print('label counts:',self._labelC, file=sys.stderr)
        
        self._featureIndexes.freeze()
    
    def _computeScore(self, featureMap, weights, labelIndex):
        '''Compute the dot product of a set of feature values and the corresponding weights.'''
        if labelIndex==-1:
            return 0.0
        
        dotProduct = 0.0
        for h,v in featureMap.items():
            dotProduct += weights[self.getGroundedFeatureIndex(h, labelIndex)]*v
        return dotProduct
    
    def _viterbi(self, sent, o0Feats, float[:] weights, float[:, :] dpValues, int[:, :] dpBackPointers, includeLossTerm=False, costAugVal=0.0, useBIO=False):
        
        nTokens = len(sent)
        
        # expand the size of dynamic programming tables if necessary
        if len(dpValues)<nTokens:
            #dpValues = [[0.0]*len(self._labels) for t in range(nTokens*1.5)]
            #dpBackPointers = [[0]*len(self._labels) for t in range(nTokens*1.5)]
            
            dpValues = cvarray(shape=(nTokens*1.5, len(self._labels)), itemsize=sizeof(float), format='f')
            dpBackPointers = cvarray(shape=(nTokens*1.5, len(self._labels)), itemsize=sizeof(int), format='i')
            
        o1FeatWeights = {l: {} for l in range(len(self._labels))}   # {current label -> {prev label -> weight}}
            
        c_viterbi(sent, o0Feats, weights, dpValues, dpBackPointers, self._labels, self._featureIndexes, o1FeatWeights, includeLossTerm, costAugVal, useBIO)
    
    def train(self, trainingData, savePrefix, averaging=False, maxIters=2, developmentMode=False, useBIO=False, includeLossTerm=False, costAugVal=0.0):
        '''Train using the perceptron. See Collins paper on discriminative HMMs.'''
        
        assert maxIters>0,maxIters
        print('training with the perceptron for up to',maxIters,'iterations', file=sys.stderr)
        
        # create feature vocabulary for the training data
        assert trainingData
        self._createFeatures(trainingData)
        
        # save features
        if developmentMode and savePrefix is not None:
            # print features before training
            with open(savePrefix+'.features', 'w') as outF:
                self.printFeatures(outF)
        
        # training iterations: calls decode()
        for i,weights in enumerate(self.decode(trainingData, maxTrainIters=maxIters, averaging=averaging, 
                                               useBIO=useBIO, includeLossTerm=includeLossTerm, costAugVal=costAugVal)):
            # store the weights in an attribute
            self._weights = weights
            
            # if dev mode, save each model and human-readable weights file
            if developmentMode:
                #self.test()
                if savePrefix is not None:
                    self.saveModel(savePrefix+'.'+str(i))
                    with open(savePrefix+'.'+str(i)+'.weights', 'w') as outF:
                        self.printWeights(outF, weights)
        
        # save model
        if savePrefix is not None:
            self.saveModel(savePrefix)
        
    
    def decode(self, data, maxTrainIters=0, averaging=False, useBIO=False, includeLossTerm=False, costAugVal=0.0):
        '''Decode a dataset under a model. Predictions are stored in the sentence within the call to _viterbi(). 
        If maxTrainIters is positive, update the weights. 
        After each iteration, the weights are yielded.'''
        
        print('decoding data type:', type(data), file=sys.stderr)
        print('averaging:',averaging,'BIO:',useBIO,'costAug:',includeLossTerm,costAugVal, file=sys.stderr)
        
        MAX_NUM_TOKENS = 200
        nLabels = len(self._labels)
        nWeights = len(self._labels)*len(self._featureIndexes)
        
        # create DP tables
        #dpValues = [[0.0]*nLabels for t in range(MAX_NUM_TOKENS)];
        #dpBackPointers = [[0]*nLabels for t in range(MAX_NUM_TOKENS)]
        dpValues = cvarray(shape=(MAX_NUM_TOKENS, nLabels), itemsize=sizeof(float), format='f')
        dpBackPointers = cvarray(shape=(MAX_NUM_TOKENS, nLabels), itemsize=sizeof(int), format='i')
        
        update = (maxTrainIters>0)   # training?
        
        # if averaging, finalWeights will contain a running average of the currentWeights vectors at all timesteps
        #finalWeights = [0.0]*nWeights
        #currentWeights = [0.0]*nWeights
        finalWeights = cvarray(shape=(nWeights,), itemsize=sizeof(float), format='f')
        currentWeights = cvarray(shape=(nWeights,), itemsize=sizeof(float), format='f')
        
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
        
        totalInstancesProcessed = 0
        
        
        
        for numIters in range(max(1,maxTrainIters)):
            if update:
                print('iter = ',numIters, file=sys.stderr)
            
            # TODO: shuffle the training data if not reading it incrementally?
            
            nWeightUpdates = 0
            
            for isent,(sent,o0Feats) in enumerate(data): # to limit the number of instances, see _createFeatures()
                
                self._viterbi(sent, o0Feats, currentWeights, dpValues, dpBackPointers,
                              includeLossTerm=False, costAugVal=0.0, useBIO=useBIO)
        
                if update:
                    nWeightUpdates += self._perceptronUpdate(sent, o0Feats, currentWeights, totalInstancesProcessed, finalWeights)
                    # will update currentWeights as well as running average in finalWeights
                
                for i in range(len(sent)):
                    if sent[i].gold != sent[i].prediction:
                        nWordsIncorrect += 1
                nWordsProcessed += len(sent)
                totalInstancesProcessed += 1
                #print(',', end='', file=sys.stderr)
                
                if isent==0:
                    print(' '.join(tkn.prediction for tkn in sent), file=sys.stderr)
                
                if totalInstancesProcessed%100==0:
                    print((finalWeights[0],finalWeights[1],finalWeights[2]),file=sys.stderr)
                    print('totalInstancesProcessed = ',totalInstancesProcessed, file=sys.stderr)
                    print('pct. correct words in last 100 inst.: {:.2%}'.format((nWordsProcessed-nWordsIncorrect)/nWordsProcessed), file=sys.stderr)
                    nWordsIncorrect = nWordsProcessed = 0
                elif totalInstancesProcessed%10==0:
                    print('.', file=sys.stderr, end='')
            
            if update and not averaging:
                finalWeights = currentWeights
                
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
            print(''.join(fname), file=out)
    
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
    def boolflag(name, description, default=False, **kwargs):
        opts.add_argument(('--' if len(name)>1 else '-')+name, action='store_false' if default else 'store_true', help=description, **kwargs)
    
    flag("train", "Path to training data feature file")
    boolflag("disk", "Load instances from the feature file in each pass through the training data, rather than keeping the full training data in memory")
    flag("iters", "Number of passes through the training data", ftype=int, default=1)
    flag("test", "Path to test data for a CoNLL-style evaluation; scores will be printed to stderr (following training, if applicable)")
    boolflag("debug", "Whether to save the list of feature names (.features file) prior to training, as well as an intermediate model (serialized model file and text file with feature weights) after each iteration of training")
    flag("labels", "List of possible labels, one label per line")
    flag("save", "Save path for serialized model file (training only). Associated output files (with --debug) will add a suffix to this path.")
    flag("load", "Path to serialized model file (decoding only)")
    flag("properties", "Properties file with option defaults", default="tagger.properties")
    #boolflag("mira"),
    boolflag("weights", "Write feature weights to stdout after training")
    flag("test-predict", "Path to feature file on which to make predictions (following training, if applicable); predictions will be written to stdout. (Will be ignored if --test is supplied.)")
    
    # formerly only allowed in properties file
    boolflag("bio", "Constrain label bigrams in decoding such that the 'O' label is never followed by a label beginning with 'I'", default=False)
    flag("costAug", "Value of cost penalty for errors against recall (for recall-oriented learning)", ftype=float, default=0.0)
    boolflag("usePrevLabel", "Include a first-order (label bigram) feature", default=True)
    
    # formerly: "useFeatureNumber"
    flag("excludeFeatures","Comma-separated list of (0-based) column numbers to ignore when reading feature files. (Do not specify column 0; use --no-lex instead.)", default='')
    
    boolflag("no-lex", "Don't include features for current and context token strings")
    boolflag("no-averaging", "Don't use averaging in perceptron training")
    
    args = opts.parse_args()
    
    if args.train is None and args.load is None:
        raise Exception('Missing argument: --train or --load')
    if args.labels is None and args.load is None:
        raise Exception('Missing argument: --labels')
    
    
    
        
    if args.load is not None:
        print('loading model from',args.load,'...', file=sys.stderr)
        t = DiscriminativeTagger.loadModel(args.load)
        # override options used during training that may be different for prediction
        #t.setBinaryFeats(False)
        print('done.', file=sys.stderr)
    else:
        t = DiscriminativeTagger()
        #t.setBinaryFeats(False)
        labels = DiscriminativeTagger.loadLabelList(args.labels)
        t._labels = labels  # TODO: "private" access
        #t._labels = ['0', 'B-noun.person', 'I-noun.person']  # TODO: debugging purposes
        
        print('training model from',args.train,'...', file=sys.stderr)
        
        if not args.disk:
            #data = DiscriminativeTagger.loadSuperSenseData(args.train, labels)
            trainingData = SupersenseFeaturizer(SupersenseDataSet(args.train, t._labels), t._featureIndexes, cache_features=True)
            
            t.train(trainingData, args.save, maxIters=args.iters, averaging=(not args.no_averaging), 
                    developmentMode=args.debug, 
                    useBIO=args.bio, includeLossTerm=(args.costAug!=0.0), costAugVal=args.costAug)
            
            del trainingData
        else:
            raise NotImplemented()
    
    
    if args.test is not None:
        #data = DiscriminativeTagger.loadSuperSenseData(args.test, t.getLabels())
        data = SupersenseFeaturizer(SupersenseDataSet(args.test, t._labels), t._featureIndexes, cache_features=False)
        
        next(t.decode(data, maxTrainIters=0, averaging=(not args.no_averaging),
                      useBIO=args.bio, includeLossTerm=(args.costAug!=0.0), costAugVal=args.costAug))
    
    elif args.weights:
        t.printWeights(sys.stdout)
    elif args.test_predict:
        t.printPredictions(args.test_predict, t.getLabels(), t.getWeights())
    else:
        t.tagStandardInput()

if __name__=='__main__':
    #import cProfile
    #cProfile.run('main()')
    try:
        main()
    except KeyboardInterrupt:
        raise
