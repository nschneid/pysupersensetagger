'''
Created on Jul 24, 2012

@author: Nathan Schneider (nschneid)
'''
from __future__ import print_function, division
import sys, codecs, random

from labeledSentence import LabeledSentence
import supersenseFeatureExtractor, morph

class DataSet(object):
    def __init__(self, f):
        self._file = f
    def __iter__(self):
        return 

class SupersenseDataSet(DataSet):
    def __init__(self, path, labels):
        self._path = path
        self._labels = labels
        self.open_file()
    
    def close_file(self):
        self._f.close()
    
    def open_file(self):
        self._f = codecs.open(self._path, 'r', 'utf-8')
    
    def __iter__(self, autoreset=True):
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
        if True:
            sent = LabeledSentence()
            for ln in self._f:
                if not ln.strip():
                    if len(sent)>0:
                        yield sent
                        sent = LabeledSentence()
                    continue
                parts = ln[:-1].split('\t')
                if len(parts)>3:
                    if parts[3]!='':
                        sent.articleId = parts[3]
                    parts = parts[:3]
                token, pos, label = parts
                label = DiscriminativeTagger.removeExtraLabels(label, self._labels)
                label = intern(str(label))
                pos = intern(str(pos))
                stemS = morph.stem(token,pos)
                sent.addToken(token=token, stem=stemS, pos=pos, goldLabel=label)
                
            if len(sent)>0:
                yield sent
                
            if autoreset:
                self.close_file()
                self.open_file()


class DiscriminativeTagger(object):
    def __init__(self):
        self._featureIndexes = supersenseFeatureExtractor.SequentialStringIndexer()
        self._trainingData = None
        self._labels = []
        self._rgen = random.Random(1234567)
        
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
    
    def printWeights(self):
        for index,fname in sorted(self._featureIndexes.items(), key=lambda x: x[1]):
            for i,label in enumerate(self._labels):
                value = self._weights[i*len(self._featureIndexes)+index]
                if value!=0.0:
                    print(label, fname, value, sep='\t')
            print()
            
    def tagStandardInput(self):
        # TODO: this depends on MaxentTagger from the Stanford tools for decoding
        pass
    
    def getGroundedFeatureIndex(self, liftedFeatureIndex, labelIndex):
        return liftedFeatureIndex + labelIndex*len(self._featureIndexes)
    
    def _perceptronUpdate(self, sent, currentWeights, timestep, runningAverageWeights):
        '''
        Update weights by iterating through the sequence, and at each token position 
        adding the feature vector for the correct label and subtracting the feature 
        vector for the predicted label.
        @param sent: the sentence
        @param currentWeights: latest value of the parameter value
        @param timestamp: number of previou updates that have been applied
        @param runningAverageWeights: average of the 'timestamp' previous weight vectors
        @return: number of weights updated
        '''
        
        if sent.predictionsAreCorrect(): return 0
        
        updates = set()
        
        for i in range(len(sent)):
            pred = self._labels.index(sent[i].prediction)
            gold = self._labels.index(sent[i].gold)
            
            if pred==gold: continue # TODO: is this correct if we are being cost-augmented?
            
            # update gold label feature weights
            
            # zero-order features
            o0FeatureMap = supersenseFeatureExtractor.extractFeatureValues(sent, i, usePredictedLabels=False, orders={0}, indexer=self._featureIndexes)
            for h,v in o0FeatureMap.items():
                featIndex = self.getGroundedFeatureIndex(h, gold)
                currentWeights[featIndex] += v
                updates.add(featIndex)
                
            # first-order features
            if supersenseFeatureExtractor.hasFirstOrderFeatures() and i>0:
                o1FeatureMap = supersenseFeatureExtractor.extractFeatureValues(sent, i, usePredictedLabels=False, orders={1}, indexer=self._featureIndexes)
                for h,v in o1FeatureMap.items():
                    featIndex = self.getGroundedFeatureIndex(h, gold)
                    currentWeights[featIndex] += v
                    updates.add(featIndex)
            
            if not o0FeatureMap and not o1FeatureMap:
                raise Exception('No features found for this token')
        
            
            # update predicted label feature weights
            
            # zero-order features
            o0FeatureMap = supersenseFeatureExtractor.extractFeatureValues(sent, i, usePredictedLabels=True, orders={0}, indexer=self._featureIndexes)
            for h,v in o0FeatureMap.items():
                featIndex = self.getGroundedFeatureIndex(h, pred)
                currentWeights[featIndex] -= v
                updates.add(featIndex)
                
            # first-order features
            if supersenseFeatureExtractor.hasFirstOrderFeatures() and i>0:
                o1FeatureMap = supersenseFeatureExtractor.extractFeatureValues(sent, i, usePredictedLabels=True, orders={1}, indexer=self._featureIndexes)
                for h,v in o1FeatureMap.items():
                    featIndex = self.getGroundedFeatureIndex(h, pred)
                    currentWeights[featIndex] -= v
                    updates.add(featIndex)
            
            if not o0FeatureMap and not o1FeatureMap:
                raise Exception('No features found for this token')
            
        for featIndex in updates:
            runningAverageWeights[featIndex] = (timestep*runningAverageWeights[featIndex] + currentWeights[featIndex])/(timestep+1)
            
        return len(updates)
    
    def _createFeatures(self, sentIndices=slice(0,10)):
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
        for nSent,sent in enumerate(self._trainingData):
            if nSent<sentIndices.start: continue
            if nSent>sentIndices.stop: break
            for i in range(len(sent)):
                # will index new features as they are encountered
                supersenseFeatureExtractor.extractFeatureValues(sent, i, usePredictedLabels=False, orders=ORDERS0, indexer=self._featureIndexes)
                '''
                for h,v in supersenseFeatureExtractor.extractFeatureValues(sent, i, usePredictedLabels=False, orders=ORDERS0, indexer=self._featureIndexes).items():
                    # TODO: first-order features handled above, so zero-order only here
                    self._featureIndexes.add(h)
                '''
                    
            print(',', file=sys.stderr, end='')
            if nSent%1000==0:
                print('.', file=sys.stderr, end='')
        
        self._trainingData.close_file()
        self._trainingData.open_file()
        
        # now create the array of feature weights
        nWeights = len(self._labels)*len(self._featureIndexes)
        
        print(' done with',nSent,'sentences:',len(self._labels),'labels,',len(self._featureIndexes),'lifted features, size',nWeights,'weight vector', file=sys.stderr)
        
        self._featureIndexes.freeze()
    
    @staticmethod
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
    
    def _computeScore(self, featureMap, weights, labelIndex):
        '''Compute the dot product of a set of feature values and the corresponding weights.'''
        if labelIndex==-1:
            return 0.0
        
        dotProduct = 0.0
        for h,v in featureMap.items():
            dotProduct += weights[self.getGroundedFeatureIndex(h, labelIndex)]*v
        return dotProduct
    
    def _viterbi(self, sent, weights, dpValues, dpBackPointers, includeLossTerm=False, costAugVal=0.0, useBIO=False):
        '''Uses the Viterbi algorithm to decode, i.e. find the best labels for the sequence 
        under the current weight vector. Updates the predicted labels in 'sent'. 
        Used in both training and testing.'''
        
        nTokens = len(sent)
        
        # expand the size of dynamic programming tables if necessary
        if len(dpValues)<nTokens:
            dpValues = [[0.0]*(nTokens*1.5) for t in range(len(self._labels))]
            dpBackPointers = [[0]*(nTokens*1.5) for t in range(len(self._labels))]
            
        prevLabel = None
        
        for i, tok in enumerate(sent):
            sent[i] = tok._replace(prediction=None)
        
        for i in range(nTokens):
            o0FeatureMap = supersenseFeatureExtractor.extractFeatureValues(sent, i, usePredictedLabels=True, orders={0}, indexer=self._featureIndexes)
            
            for l,label in enumerate(self._labels):
                maxScore = float('-inf')
                maxIndex = -1
                
                # score for zero-order features
                score0 = self._computeScore(o0FeatureMap, weights, l)
                
                # cost-augmented decoding
                if label!=sent[i].gold:
                    if includeLossTerm:
                        score0 += 1.0   # base cost of any error
                    if label=='O':
                        score0 += costAugVal    # recall-oriented penalty (for erroneously predicting 'O')

                # consider each possible previous label
                for k,prevLabel in enumerate(self._labels):
                    if not self.legalTagBigram(None if i==0 else prevLabel, label, useBIO):
                        continue
                    
                    # compute correct score based on previou scores
                    score = 0.0
                    if i>0:
                        sent[i-1] = sent[i-1]._replace(prediction=prevLabel)
                        score = dpValues[i-1][k]
                    
                    # the score for the previou label is added on separately here,
                    # in order to avoid computing the whole score--which only 
                    # depends on the previous label for one feature--a quadratic 
                    # number of times
                    # TODO: plus vs. times doesn't matter here, right? use plus to avoid numeric overflow
                    
                    # score of moving from label k at the previous position to the current position (i) and label (l)
                    score += score0
                    if supersenseFeatureExtractor.hasFirstOrderFeatures() and i>0:
                        o1FeatureMap = supersenseFeatureExtractor.extractFeatureValues(sent, i, usePredictedLabels=True, orders={1}, indexer=self._featureIndexes)
                        for h,v in o1FeatureMap.items():
                            score += weights[self.getGroundedFeatureIndex(h, l)]*v
                            
                    # find the max of the combined score at the current position
                    # and store the backpointer accordingly
                    if score>maxScore:
                        maxScore = score
                        maxIndex = k
                    
                    # if this is the first token, there is only one possible 
                    # previous label
                    if i==0:
                        break
                    
                dpValues[i][l] = maxScore
                dpBackPointers[i][l] = maxIndex
        
        # decode from the lattice
        # extract predictions from backpointers
        
        # first, find the best label for the last token
        maxIndex, maxScore = max(enumerate(dpValues[nTokens-1]), key=lambda x: x[1])
        
        # now proceed backwards, following backpointers
        for i in range(nTokens-1,-1,-1):
            sent[i] = sent[i]._replace(prediction=self._labels[maxIndex])
            maxIndex = dpBackPointers[i][maxIndex]
    
    
    def train(self, savePrefix, averaging=False, maxIters=5, developmentMode=False, maxInstances=10):
        '''Train using the perceptron. See Collins paper on discriminative HMMs.'''
        
        assert self._trainingData
        
        print('training with the perceptron', file=sys.stderr)
        
        # create DP tables
        MAX_NUM_TOKENS = 200
        nLabels = len(self._labels)
        dpValues = [[0.0]*nLabels for t in range(MAX_NUM_TOKENS)];
        dpBackPointers = [[0]*nLabels for t in range(MAX_NUM_TOKENS)]
        
        self._createFeatures()
        nWeights = len(self._labels)*len(self._featureIndexes)
        
        print('training data type:', type(self._trainingData), file=sys.stderr)
        
        # finalWeights will contain a running average of the currentWeights vectors at all timesteps
        finalWeights = [0.0]*nWeights
        currentWeights = [0.0]*nWeights
        
        # tabulate accuracy at every 500 iterations
        nWordsProcessed = 0
        nWordsIncorrect = 0
        
        totalInstancesProcessed = 0
        
        if developmentMode and savePrefix is not None:
            # print features before training
            with open(savePrefix+'.features', 'w') as outF:
                self.printFeatures(outF)
            
        for numIters in range(maxIters):
            print('iter = ',numIters, file=sys.stderr)
            
            # TODO: shuffle the training data if not reading it incrementally?
            
            nWeightUpdates = 0
            
            for sent in self._trainingData:
                self._viterbi(sent, currentWeights, dpValues, dpBackPointers)
                nWeightUpdates += self._perceptronUpdate(sent, currentWeights, totalInstancesProcessed, finalWeights)
                # will update currentWeights as well as running average in finalWeights
                
                for i in range(len(sent)):
                    if sent[i].gold != sent[i].prediction:
                        nWordsIncorrect += 1
                nWordsProcessed += len(sent)
                totalInstancesProcessed += 1
                print(',', end='', file=sys.stderr)
                
                if totalInstancesProcessed%500==0:
                    print('totalInstancesProcessed = ',totalInstancesProcessed, file=sys.stderr)
                    print('pct. correct words in last 500 inst.: {:.2%}'.format((nWordsProcessed-nWordsIncorrect)/nWordsProcessed))
                    nWordsIncorrect = nWordsProcessed = 0
                
                if totalInstancesProcessed==maxInstances:
                    break
                
            if developmentMode:
                #self.test()
                
                if savePrefix is not None:
                    if not averaging:
                        finalWeights = currentWeights
                    
                    self.saveModel(savePrefix+'.'+str(numIters))
                    with open(savePrefix+'.'+str(numIters)+'.weights', 'w') as outF:
                        self.printWeights(outF, currentWeights)
                
            print('weight updates this iteration:',nWeightUpdates, file=sys.stderr)
            if nWeightUpdates==0:
                print('converged! stopped training', file=sys.stderr)
                break
            
        if not averaging:
            finalWeights = currentWeights
            
        if savePrefix is not None:
            self.saveModel(savePrefix)
            
        return finalWeights
        
    def printFeatures(self, out):
        print(len(self._featureIndexes),'lifted features x',len(self._labels),'labels =',len(self._featureIndexes)*len(self._labels),'grounded features', file=out)
        print('labels:',self._labels,'\n', file=out)
        for fname in sorted(self._featureIndexes.strings):
            print(''.join(fname), file=out)
    
    def saveModel(self, savePrefix):
        raise NotImplemented()
    
    def test(self, weights):
        raise NotImplemented()
        '''
        if self._testData is None: return
        
        for sent in self._testData:
            self._viterbi(sent, weights, dpValues, dpBackPointers, includeLossTerm, costAugVal, useBIO)
        
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
    boolflag("useBIO", "Constrain label bigrams in decoding such that the 'O' label is never followed by a label beginning with 'I'", default=True)
    flag("useCostAug", "Value of cost penalty for errors against recall (for recall-oriented learning)", ftype=float, default=0.0)
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
        print('training model from',args.train,'...', file=sys.stderr)
        t = DiscriminativeTagger()
        #t.setBinaryFeats(False)
        labels = DiscriminativeTagger.loadLabelList(args.labels)
        t._labels = labels  # TODO: "private" access
        t._labels = ['0', 'B-noun.person', 'I-noun.person']  # TODO: debugging purposes
        
        if not args.disk:
            #data = DiscriminativeTagger.loadSuperSenseData(args.train, labels)
            data = SupersenseDataSet(args.train, t._labels)
            t._trainingData = data  # TODO: "private" access
        else:
            raise NotImplemented()
        
    if args.test is not None:
        #data = DiscriminativeTagger.loadSuperSenseData(args.test, t.getLabels())
        data = SupersenseDataSet(args.test, t._labels)
        t.setTestData(data)
    if args.load is None:
        t._weights = t.train(args.save, maxIters=args.iters, developmentMode=args.debug)
        
    if args.test is not None:
        t.test()
    elif args.weights:
        t.printWeights(sys.stdout)
    elif args.test_predict:
        t.printPredictions(args.test_predict, t.getLabels(), t.getWeights())
    else:
        t.tagStandardInput()

if __name__=='__main__':
    #import cProfile
    #cProfile.run('main()')
    main()
 
