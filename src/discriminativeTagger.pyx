# cython: profile=True
# cython: infer_types=True
'''
Created on Jul 24, 2012

@author: Nathan Schneider (nschneid)
'''
from __future__ import print_function, division
import sys, codecs, random, math
from numbers import Number
from collections import defaultdict, Counter

cimport cython
from cython.view cimport array as cvarray

from labeledSentence import LabeledSentence
import morph

from pyutil.ds import features 

import supersenseFeatureExtractor
featureExtractor = None


from dataFeaturizer import SupersenseDataSet, SupersenseTrainSet, SupersenseFeaturizer

from decoding cimport _ground0, _ground, c_viterbi, i_viterbi

# inline functions _ground0() and _ground() are duplicated in decoding.pyx


cdef class Weights(object):
    cdef float _l0, _l1, _sql2
    cdef list _w
    def __init__(self, sizeOrDefault=None):
        cdef int percept, i
        cdef dict d
        self._w = []
        self._l0 = 0
        self._l1 = 0
        self._sql2 = 0
        if sizeOrDefault:   # size (number of percepts)
            if isinstance(sizeOrDefault,int):
                self._w = [dict() for i in range(sizeOrDefault)]
            else: # copy constructor
                for percept,d in enumerate(sizeOrDefault._w):
                    self._w[percept] = dict(d)
                self._l0 = sizeOrDefault._l0
                self._l1 = sizeOrDefault._l1
                self._sql2 = sizeOrDefault._sql2
    
    def p(self, int percept):
        return self._w[percept]
    
    def nPercepts(self):
        return len(self._w)
    
    def __getitem__(self, key):
        cdef int percept, label
        percept, label = key
        return self._w[percept].get(label,0.0)
    
    def __setitem__(self, key, float value):
        cdef int percept, label
        cdef float oldvalue
        cdef dict perceptwts
        
        percept, label = key
        perceptwts = self._w[percept]
        oldvalue = perceptwts.get(label,0.0)
        perceptwts[label] = value
        
        if oldvalue!=0.0 and value==0.0:
            self._l0 -= 1
        elif oldvalue==0.0 and value!=0.0:
            self._l0 += 1
        self._l1 -= abs(oldvalue)
        self._sql2 -= oldvalue*oldvalue
        self._l1 += abs(value)
        self._sql2 += value*value
        
    def store_average(self, Weights currentWeights, Weights avgWeightDeltas, int timestep):
        for percept in range(currentWeights.nPercepts()):
            for l in set(currentWeights._w[percept]) | set(avgWeightDeltas._w[percept]):
                self[percept,l] = currentWeights[percept,l] - avgWeightDeltas[percept,l]/timestep
                

class ArrayWeights(object):
    def __init__(self, sizeOrDefault):
        self._w = []
        self._l0 = 0
        self._l1 = 0
        self._sql2 = 0
        if sizeOrDefault:   # size (number of percepts)
            if isinstance(sizeOrDefault,tuple):
                nPercepts, nLabels = sizeOrDefault
                self._w = [0.0]*(nPercepts*nLabels)
                self._nPercepts = nPercepts
                self._nLabels = nLabels
            else: # copy constructor
                self._w = list(sizeOrDefault)
                self._l0 = sizeOrDefault._l0
                self._l1 = sizeOrDefault._l1
                self._sql2 = sizeOrDefault._sql2
                self._nPercepts = sizeOrDefault._nPercepts
                self._nLabels = sizeOrDefault._nLabels
    
    def p(self, percept): # labels and weights for percept
        return {l: self[percept,l] for l in range(self._nLabels)}
    
    def __getitem__(self, key):
        percept, label = key
        return self._w[_ground0(percept, label, self._nPercepts)]
    
    def __setitem__(self, key, value):
        percept, label = key
        featIndex = _ground0(percept, label, self._nPercepts)
        oldvalue = self._w[featIndex]
        self._w[featIndex] = value
        if oldvalue!=0.0 and value==0.0:
            self._l0 -= 1
        elif oldvalue==0.0 and value!=0.0:
            self._l0 += 1
        self._l1 -= abs(oldvalue)
        self._l1 += abs(value)
        self._sql2 -= oldvalue*oldvalue
        self._sql2 += value*value
        
    def store_average(self, currentWeights, avgWeightDeltas, timestep):
        for i in range(len(self._w)):
            self._w[i] = currentWeights._w[i] - avgWeightDeltas._w[i]/timestep

'''
cdef float l2norm(float[:] weights):
    cdef float t
    cdef int i
    t = 0.0
    for i in range(weights.shape[0]):
        t += weights[i]*weights[i]
    return t**0.5
'''
cdef l2norm(Weights weights): return weights._sql2**0.5
"""
cdef void average_weights(object finalWeights, object currentWeights, object avgWeightDeltas, int timestep):
    '''Final step of weight averaging. See Hal Daume's thesis, Figure 2.3.'''
    cdef int i  # feature index
    for i in range(finalWeights.shape[0]):
        finalWeights[i] = currentWeights[i] - avgWeightDeltas[i]/timestep
"""
def average_weights(finalWeights, currentWeights, avgWeightDeltas, timestep):
    return finalWeights.store_average(currentWeights, avgWeightDeltas, timestep)
    

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
                ###baseline = weights[_ground0(index,d,indexerSize)]
                baseline = weights[index,d]
            for i,label in enumerate(self._labels):
                ###value = weights[_ground0(index,i,indexerSize)]
                value = weights[index,i]
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
    
    def _perceptronUpdate(self, sent, goldDerivation, predDerivation, object currentWeights, timestep, avgWeightDeltas, learningRate=1.0, sumsqgrads=None):
        '''
        Update weights by iterating through the sequence, and at each token position 
        adding the feature vector for the correct label and subtracting the feature 
        vector for the predicted label.
        @param sent: the sentence, including gold and predicted tags
        @param o0Feats: active lifted zero-order features for each token
        @param currentWeights: latest value of the parameter value
        @param timestep: number of training instances seen so far, including this one (always positive)
        @param avgWeightDeltas: number of iterations times the difference between the current weight vector and the weight averages
        (for efficient averaging as described in Hal Daume's thesis, Figure 2.3)
        @return: number of weights updated
        '''
        
        assert timestep>0
        
        if sent.predictionsAreCorrect(): return 0
        
        updates = set()    # indices of weights updated
        
        cdef int h, i, goldLabel, predLabel
        cdef float v
        
        for indices,factorFeats in goldDerivation:
            
            if not any(sent[i].prediction!=sent[i].gold for i in indices): continue
            # is this correct if we are being cost-augmented?
            # yes, so long as the decoder used the cost properly to determine the prediction
            
            goldLabel = self._labels.index(sent[indices[-1]].gold) if len(self._labels)>1 else 0
            
            # update gold label feature weights
            
            # FOR NOW, we assume no features are shared between classes.
            # Otherwise we may have to ensure the learner isn't penalizing features that 
            # should fire for both the gold and predicted classes.
            
            '''
            AdaGrad update chooses a different step size for each parameter.
            Green et al. 2013, eqs. (4) and (5):
            With respect to a single parameter (component) j:
                g_t = gradient w.r.t. j of the loss under the current weight vector
                G_t = G_{t-1} + math.pow(g_t, 2)    # i.e., G_t is the sum of squared gradients for this feature until now
                w_t = w_{t-1} - η * math.pow(G_t, -0.5) * g_t
            We fix η = 1.
            '''
            
            for h,vv in factorFeats.items():
                ###featIndex = _ground(h, goldLabel, self._featureIndexes)
                featIndex = (h,goldLabel)
                v = vv(self._labels[goldLabel]) if not isinstance(vv,Number) else vv
                updates.add(featIndex)
                if sumsqgrads:
                    sumsqgrads[featIndex] += v*v
                    learningRate = sumsqgrads[featIndex]**(-0.5)
                currentWeights[featIndex] += learningRate * v
                avgWeightDeltas[featIndex] += timestep * learningRate * v
            
            if not factorFeats:
                raise Exception('No features found for this token')
        
            
        for indices,factorFeats in predDerivation:
            
            if not any(sent[i].prediction!=sent[i].gold for i in indices): continue
            # is this correct if we are being cost-augmented?
            # yes, so long as the decoder used the cost properly to determine the prediction
            
            predLabel = self._labels.index(sent[indices[-1]].prediction) if len(self._labels)>1 else 0
            
            # update predicted label feature weights
            
            for h,vv in factorFeats.items():
                ###featIndex = _ground(h, predLabel, self._featureIndexes)
                featIndex = (h,predLabel)
                v = vv(self._labels[predLabel]) if not isinstance(vv,Number) else vv
                updates.add(featIndex)
                if sumsqgrads:
                    sumsqgrads[featIndex] += v*v
                    learningRate = sumsqgrads[featIndex]**(-0.5)
                currentWeights[featIndex] -= learningRate * v
                avgWeightDeltas[featIndex] -= timestep * learningRate * v
            
            if not factorFeats:
                raise Exception('No features found for this token')
        
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
        for h,v in featureMap.iteritems():
            dotProduct += weights[self.getGroundedFeatureIndex(h, labelIndex)]*v
        return dotProduct
    
    def _viterbi(self, nLabels, weights, 
                 includeLossTerm=False, costAugVal=0.0, useBIO=False):
        
        # setup
        MAX_NUM_TOKENS = 200
        o0Scores = cvarray(shape=(MAX_NUM_TOKENS, nLabels), itemsize=sizeof(float), format='f')
        dpValuesFwd = cvarray(shape=(MAX_NUM_TOKENS, nLabels), itemsize=sizeof(float), format='f')
        dpValuesBwd = cvarray(shape=(MAX_NUM_TOKENS, nLabels), itemsize=sizeof(float), format='f')
        dpBackPointers = cvarray(shape=(MAX_NUM_TOKENS, nLabels), itemsize=sizeof(int), format='i')
        
        nDegenerateLabels = int(math.ceil(math.log(nLabels,2)))  # for iterative Viterbi
        o1FeatWeights = cvarray(shape=(2, nLabels+nDegenerateLabels, nLabels+nDegenerateLabels), itemsize=sizeof(float), format='f')
        o1FeatWeights[:,:,:] = float('inf')    # INF = uninitialized. note that the contents do not depend on the sentence.
        
        labelScores0 = cvarray(shape=(len(self._labels),), itemsize=sizeof(float), format='f')
        
        ##########################
        # for each instance
        
        instance = (yield o1FeatWeights) # pass back o1FeatWeights and receive first instance from send()
        
        while True:
            
            sent,o0Feats = instance
            
            nTokens = len(sent)
            
            
            # expand the size of dynamic programming tables if necessary
            if dpValuesFwd.shape[0]<nTokens:
                #dpValues = [[0.0]*len(self._labels) for t in range(int(nTokens*1.5))]
                #dpBackPointers = [[0]*len(self._labels) for t in range(int(nTokens*1.5))]
                dpValuesFwd = cvarray(shape=(int(nTokens*1.5), len(self._labels)), itemsize=sizeof(float), format='f')
                dpValuesBwd = cvarray(shape=(int(nTokens*1.5), len(self._labels)), itemsize=sizeof(float), format='f')
                dpBackPointers = cvarray(shape=(int(nTokens*1.5), len(self._labels)), itemsize=sizeof(int), format='i')
                o0Scores = cvarray(shape=(int(nTokens*1.5), len(self._labels)), itemsize=sizeof(float), format='f')
            
            METHOD = 'c'   # conventional and/or iterative Viterbi
            if 'c' in METHOD:
                c_score, derivation = c_viterbi(sent, o0Feats, featureExtractor, weights, dpValuesFwd, dpBackPointers, labelScores0, o1FeatWeights, self._labels, self._featureIndexes, includeLossTerm, costAugVal, useBIO)
                c_preds = [x.prediction for x in sent]
            if 'i' in METHOD:
                i_score, derivation = i_viterbi(sent, o0Feats, featureExtractor, weights, dpValuesFwd, dpValuesBwd, dpBackPointers, o0Scores, o1FeatWeights, self._labels, self._freqSortedLabelIndices, self._featureIndexes, includeLossTerm, costAugVal, useBIO)
                i_preds = [x.prediction for x in sent]
            if 'c' in METHOD and 'i' in METHOD: # check that results match
                print(c_score,c_preds)
                print(i_score,i_preds)
                print('---')
                assert c_score==i_score,(c_score,i_score)
            
            # yield back the instance, which now contains predictions, and receive the next instance
            instance = (yield (sent, derivation))
        
        # no tear-down necessary

    def train(self, trainingData, savePrefix, instanceIndices=None, averaging=False, 
              tuningData=None, earlyStopInterval=None, earlyStopDelay=0, 
              maxIters=2, developmentMode=False, useBIO=False, includeLossTerm=False, costAugVal=0.0, gamma=1.0, gammaUpdate=None and 'adagrad'):
        '''Train using the perceptron. See Collins paper on discriminative HMMs.'''
        assert maxIters>0,maxIters
        assert earlyStopInterval is None or tuningData is not None
        print('training with the perceptron for up to',maxIters,'iterations', 
              ('with early stopping by checking the tuning data every {} iterations'.format(abs(earlyStopInterval)) if earlyStopInterval is not None else ''),
              file=sys.stderr)
        if gamma!=1.0:
            print('gamma (step size) =',gamma, file=sys.stderr)
        if gammaUpdate:
            print('gammaUpdate:',gammaUpdate, file=sys.stderr)
            assert gammaUpdate=='adagrad' 
        # create feature vocabulary for the training data
        assert trainingData
        self._createFeatures(trainingData, sentIndices=instanceIndices)
        trainingData.enable_caching()   # don't cache before the featureset is finalized!
        
        # save features
        if developmentMode and savePrefix is not None:
            # print features before training
            with open(savePrefix+'.features', 'w') as outF:
                self.printFeatures(outF)
        
        # for best model achieved so far
        prevNCorrect = -1
        prevTotCost = float('inf')
        prevBestIter = prevWeights = None
        
        nTuning = None
        
        sumsqgrads = [] if gammaUpdate=='adagrad' else None
        
        # training iterations: calls decode()
        for i,weights in enumerate(self.learn(trainingData, maxTrainIters=maxIters, averaging=averaging, 
                                               useBIO=useBIO, includeLossTerm=includeLossTerm, costAugVal=costAugVal, gamma=gamma, sumsqgrads=sumsqgrads)):
            
            # store the new weights in an attribute
            self._weights = weights
            print('l2(prevWeights) = {:.4}, l2(weights) = {:.4}'.format(l2norm(prevWeights) if prevWeights else 0.0,l2norm(weights)), file=sys.stderr)
            
            # if dev mode, save each model and human-readable weights file
            if developmentMode:
                #self.test()
                if savePrefix is not None:
                    self.saveModel(savePrefix+'.'+str(i))
                    with open(savePrefix+'.'+str(i)+'.weights', 'w') as outF:
                        self.printWeights(outF, weights)
            
            if earlyStopInterval is not None and i<maxIters-1 and (i+1)%abs(earlyStopInterval)==0:
                # decode on tuning data and decide whether to stop
                self.decode_dataset(tuningData, print_predictions=False, 
                                    useBIO=useBIO, includeLossTerm=False, costAugVal=0.0)
                
                totCost = nCorrect = nTuning = 0
                for sent,o0Feats in tuningData:
                    nCorrect += sum(1 for tok in sent if tok.gold==tok.prediction)
                    totCost += sum(1+(costAugVal if (tok.gold=='O' or tok.gold=='o') else 0) for tok in sent if tok.gold!=tok.prediction)
                    nTuning += len(sent)
                if prevNCorrect is not None:
                    # use accuracy or cost as criterion
                    isImproved = (nCorrect > prevNCorrect) if earlyStopInterval>0 else (totCost < prevTotCost)
                    
                    if isImproved:
                        prevNCorrect = nCorrect
                        prevTotCost = totCost
                        prevBestIter = i
                        # hold on to the previous weights
                        prevWeights = self._weights.copy()
                    else:
                        if i-prevBestIter>earlyStopDelay:
                            print('stopping early after iteration',i,'; using model from iteration',prevBestIter,
                                  'instead. new tuning set acc {}/{}={:.2%}, previously {}/{}={:.2%}'.format(nCorrect,nTuning,nCorrect/nTuning,
                                                                                                      prevNCorrect,nTuning,prevNCorrect/nTuning),
                                  file=sys.stderr)
                            self._weights = prevWeights # the last model that posted an improvement
                            break                
                
            
        
        # save model
        if savePrefix is not None:
            self.saveModel(savePrefix)
    
    
    def learn(self, data, maxTrainIters, averaging=False, useBIO=False, includeLossTerm=False, costAugVal=0.0, gamma=1.0, sumsqgrads=None):
        '''Decode a dataset under a model. Predictions are stored in the sentence within the call to _viterbi(). 
        If maxTrainIters is positive, update the weights. 
        After each iteration, the weights are yielded.'''
        
        print('decoding data type:', type(data), file=sys.stderr)
        print('learning:',bool(maxTrainIters),'averaging:',averaging,'BIO:',useBIO,'costAug:',includeLossTerm,costAugVal, file=sys.stderr)
        
        
        nLabels = len(self._labels) # number of (actual) labels
        nPercepts = len(self._featureIndexes)
        nWeights = nLabels*nPercepts
        
        if not sumsqgrads and sumsqgrads is not None:   # for AdaGrad
            sumsqgrads.extend([0.0 for j in range(nWeights)])
        
        
        assert maxTrainIters>0
        
        #finalWeights = [0.0]*nWeights
        #currentWeights = [0.0]*nWeights
        
        # the model to decode with. if learning with averaging, contains the latest non-averaged weight vector. 
        # otherwise same as finalWeights.
        ###currentWeights = cvarray(shape=(nWeights,), itemsize=sizeof(float), format='f')
        ###currentWeights[:] = 0
        # the model to be written to disk. will be yielded after each iteration. includes averaging if applicable.
        ###finalWeights = cvarray(shape=(nWeights,), itemsize=sizeof(float), format='f')
        # currentWeights & finalWeights are initialized to self._weights if set, otherwise 0
        ###avgWeightDeltas = cvarray(shape=(nWeights,), itemsize=sizeof(float), format='f')
        ###avgWeightDeltas[:] = 0
        avgWeightDeltas = Weights(nPercepts)
        # initialized to 0
        
        '''
        yield finalWeights  # TODO: debugging: don't train at all!
        return
        '''
        
        if self._weights is not None:  # initialize weights
            #assert len(self._weights)==nWeights    # can't len() an array?
            ###for i,w in enumerate(self._weights):
            ###    finalWeights[i] = currentWeights[i] = w
            currentWeights = Weights(self._weights)
            finalWeights = Weights(self._weights)
        else:
            currentWeights = Weights(nPercepts)
            finalWeights = Weights(nPercepts)
        
        
        totalInstancesProcessed = 0
        
        #gamma_current = 1.0   # current learning rate
        gamma_current = gamma
        
        decoder = self.decode(nLabels, currentWeights, 
                              includeLossTerm=includeLossTerm, costAugVal=costAugVal, 
                              useBIO=useBIO)    # per-sentence decoder (persists across iterations)
        
        o1FeatWeights = decoder.next()
        
        for numIters in range(max(1,maxTrainIters)):
            print('iter = ',numIters, file=sys.stderr)
            
            # TODO: shuffle the training data if not reading it incrementally?
            
            nWeightUpdates = 0
            
            for isent,(sent,o0Feats) in enumerate(data): # to limit the number of instances, see _createFeatures()
                sent,predDerivation = decoder.send((sent,o0Feats))    # Viterbi decode this instance
                goldDerivation = [((i,), o0Feats[i]) for i in range(len(sent))]
                if featureExtractor.hasFirstOrderFeatures():
                    goldDerivation.extend([((i-1,i), {self._featureIndexes['prevLabel=',sent[i-1].gold]: 1}) for i in range(1,len(sent))])
                #print(sent, file=sys.stderr)
                #print(o0Feats, file=sys.stderr)
                #print(predDerivation, file=sys.stderr)
                #print(goldDerivation, file=sys.stderr)
                #assert False
                #gamma_current *= gamma    # keep learning rate/instance weight fixed for now
                nWeightUpdates += self._perceptronUpdate(sent, goldDerivation, predDerivation, currentWeights, totalInstancesProcessed+1, avgWeightDeltas, learningRate=gamma_current, sumsqgrads=sumsqgrads)
                # will update currentWeights as well as distance to running average in avgWeightDeltas
                o1FeatWeights[:,:,:] = float('inf') # clear the bigram feature weights cache (they will be recomputed the next time we decode)
                
                totalInstancesProcessed += 1
            
            decoder.next()  # end of a pass--tell the decoder to print summary statistics
            
            if averaging:   # compute the averages from the deltas
                print('averaging...', end='', file=sys.stderr)
                average_weights(finalWeights, currentWeights, avgWeightDeltas, totalInstancesProcessed+1)
                # average includes the initial (0) weight vector
                print('done', file=sys.stderr)
            else:
                finalWeights = currentWeights
            
            print('l2(currentWeights) = {:.4}, l2(finalWeights) = {:.4}'.format(l2norm(currentWeights), l2norm(finalWeights)), file=sys.stderr)
            
            #assert False    # TODO: debug
            
            yield finalWeights
            
            print('weight updates this iteration:',nWeightUpdates, file=sys.stderr)
            if nWeightUpdates==0:
                print('converged! stopped training', file=sys.stderr)
                break
            
        # really just a formality, I think
        decoder.close()
        
    def decode(self, nLabels, currentWeights, includeLossTerm, costAugVal, useBIO):
        '''
        Coroutine that decodes an instance (by calling _viterbi()), maintaining 
        summary statistics over all instances decoded and printing them periodically 
        to stderr.
        
        First next(): yields o1FeatWeights
        Thereafter until closed:
         - if sent None (a.k.a. next()), concludes that a pass through the data has been 
           completed and prints summary statistics
         - otherwise, treats the sent object as an instance, decodes it and yields it back 
           (now including predictions & derivation)
        '''
        
        # setup

        # tabulate accuracy at every 500 iterations
        nWordsProcessed = 0
        nWordsIncorrect = 0
        totalWordsProcessed = 0
        totalWordsIncorrect = 0
        totalInstancesProcessed = 0
        
        firstInPass = True
        reportAcc = True
        
        decoder = self._viterbi(nLabels, currentWeights, 
                              includeLossTerm=includeLossTerm, costAugVal=costAugVal, 
                              useBIO=useBIO)
        
        o1FeatWeights = decoder.next()
        
        try:
            instance = (yield o1FeatWeights)    # send back o1FeatWeights and receive the first instance
            
            while True:
                if instance is None: # signal to print accuracy and reset firstInPass = True
                    print('word accuracy over {} words in {} instances: {:.2%}'.format(totalWordsProcessed, totalInstancesProcessed, (totalWordsProcessed-totalWordsIncorrect)/totalWordsProcessed), file=sys.stderr)
                    firstInPass = True
                    instance = (yield)
                    continue
                
                sent,o0Feats = instance
                sent,derivation = decoder.send((sent,o0Feats))    # Viterbi decode this instance
                
                if reportAcc:
                    for i in range(len(sent)):
                        if sent[i].gold is None:
                            reportAcc = False
                            break
                        if sent[i].gold != sent[i].prediction:
                            nWordsIncorrect += 1
                            totalWordsIncorrect += 1
                nWordsProcessed += len(sent)
                totalWordsProcessed += len(sent)
                totalInstancesProcessed += 1
                #print(',', end='', file=sys.stderr) # DEBUG
                
                if firstInPass: # print the tagging of the first sentence in the dataset
                    print(' '.join(tkn.prediction for tkn in sent).encode('utf-8'), file=sys.stderr)
                    firstInPass = False
                
                if totalInstancesProcessed%100==0:
                    print('totalInstancesProcessed = ',totalInstancesProcessed, file=sys.stderr)
                    if reportAcc:
                        print('word accuracy in last 100 instances: {:.2%}'.format((nWordsProcessed-nWordsIncorrect)/nWordsProcessed), file=sys.stderr)
                    nWordsIncorrect = nWordsProcessed = 0
                elif totalInstancesProcessed%10==0:
                    print('.', file=sys.stderr, end='')
                
                
                instance = (yield (sent, derivation)) # next instance
                
        except GeneratorExit:   # cleanup
            # really just a formality, I think
            decoder.close()


    def decode_dataset(self, dataset, print_predictions, useBIO, includeLossTerm, costAugVal):
        '''
        Make a decoding pass through a dataset under the current model.
        Not used for the training data: see learn()
        '''
        
        nLabels = len(self._labels) # number of (actual) labels
        
        assert self._weights is not None  # model weights
        
        
        decoder = self.decode(nLabels, self._weights,
                              useBIO=useBIO, includeLossTerm=includeLossTerm, costAugVal=costAugVal)
        decoder.next()
        
        for sent,o0Feats in dataset:
            sent,derivation = decoder.send((sent,o0Feats))
            if print_predictions:
                # print predictions
                print(sent)
                print()
                
        decoder.next()  # show summary statistics
        decoder.close() # a formality

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
        if not isinstance(self._weights, (list,Weights,ArrayWeights)):
            self._weights = list(weights)
        with open(saveFP, 'wb') as saveF:
            cPickle.dump(self, saveF, protocol=2)
        self._weights = weights
    
    @staticmethod
    def loadModel(savePrefix):
        import cPickle
        saveFP = savePrefix+'.pickle'
        with open(saveFP, 'rb') as saveF:
            model = cPickle.load(saveF)
        # convert list of weights back to array
        nWeights = len(model._weights)
        weights = cvarray(shape=(nWeights,), itemsize=sizeof(float), format='f')
        for i,w in enumerate(model._weights):
            weights[i] = w
        model._weights = weights
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
    flag("early-stop-delay", "Number of extra iterations (epochs) to wait after the model has failed to improve before stopping and using the previous best model.", ftype=int, default=0)
    inflag("test", "Path to test data for a CoNLL-style evaluation; scores will be printed to stderr (following training, if applicable). (Will be ignored if --test-predict is supplied.)")
    #flag("max-test-instances", "During testing, truncate the data to the specified number of instances", ftype=int)
    boolflag("debug", "Whether to save the list of feature names (.features file) prior to training, as well as an intermediate model (serialized model file and text file with feature weights) after each iteration of training")
    inflag("YY", "List of possible labels, one label per line") # formerly: labels
    flag("defaultY", "Default (background) label: model weights represent deviation from this label")
    flag("save", "Save path for serialized model file (training only). Associated output files (with --debug) will add a suffix to this path.")
    flag("load", "Path to serialized model file (decoding only)")   #inflag
    #inflag("properties", "Properties file with option defaults", default="tagger.properties")
    #boolflag("mira"),
    boolflag("weights", "Write feature weights to stdout after training")
    flag("test-predict", "Path to test data for a CoNLL-style evaluation; predictions will be printed to stdout and scores will be printed to stderr (following training, if applicable). (Supersedes --test.)")
    flag("predict", "Path to data on which to make predictions (following training, if applicable); predictions will be written to stdout (following --test-predict predictions, if both flags are specified). (The data need not have gold labels.)")
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
    #flag("gamma","Base of learning rate (exponent is number of instances seen so far)", ftype=float, default=1.0)
    flag("gamma","Instance weight/constant learning rate", ftype=float, default=1.0)
    boolflag("adagrad","Adaptive gradient method (learning rate update)")
    boolflag("no-lex", "Don't include features for current and context token strings")
    boolflag("no-averaging", "Don't use averaging in perceptron training")
    
    # features
    boolflag("mwe", "Multiword expressions featureset")
    boolflag("no-bigrams", "Disable token bigram features", default=False)
    boolflag("no-oov", "Disable WordNet unigram OOV features.", default=False)
    boolflag("no-compound", "Disable WordNet compound features; redundant if --no-wordnet is specified", default=False)
    boolflag("cxt-pos-filter", "Filter bigram features based on the POS pairs")
    boolflag("clusters", "Word cluster features")
    flag("cluster-file", "Path to file with word clusters", default=supersenseFeatureExtractor._options['clusterFile'])
    boolflag("pos-neighbors", "POS neighbor features")
    
    inflag("lex", "Lexicons to load for lookup features", nargs='*')
    inflag("clist", "Collocation lists (ranked) to load for lookup features", nargs='*')
    
    args = opts.parse_args()
    
    if args.train is None and args.load is None:
        raise Exception('Missing argument: --train or --load')
    if args.YY is None and args.load is None:
        raise Exception('Missing argument: --YY')
    
    global featureExtractor
    if args.mwe:
        import mweFeatures as featureExtractor
    else:
        import sstFeatures as featureExtractor
    
    featureExtractor.registerOpts(args)
    
    evalData = None
    
    # load or train a model
    
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
            trainingData = SupersenseFeaturizer(featureExtractor, SupersenseTrainSet(args.train, t._labels, legacy0=args.legacy0), t._featureIndexes, cache_features=False)
            if args.test_predict is not None or args.test is not None:
                # keep labeled test data in memory so it can be used for early stopping (tuning)
                evalData = SupersenseFeaturizer(featureExtractor, SupersenseTrainSet(args.test_predict or args.test, 
                                                                                    t._labels, legacy0=args.legacy0,
                                                                                    keep_in_memory=True), 
                                                t._featureIndexes, cache_features=False)
                
            t.train(trainingData, args.save, maxIters=args.iters, instanceIndices=slice(0,args.max_train_instances), averaging=(not args.no_averaging), 
                    earlyStopInterval=args.early_stop if (args.test or args.test_predict) else None, 
                    earlyStopDelay=args.early_stop_delay if (args.test or args.test_predict) else None,
                    tuningData=evalData,
                    developmentMode=args.debug, 
                    useBIO=args.bio, includeLossTerm=args.includeLossTerm, costAugVal=args.costAug, gamma=args.gamma, gammaUpdate='adagrad' if args.adagrad else None)
            
            del trainingData
        else:
            raise NotImplemented()
    
    
    if args.test_predict is not None or args.test is not None:
        # evaluate (test), and possibly print predictions for that data
        
        if evalData is None:
            evalData = SupersenseFeaturizer(featureExtractor, SupersenseTrainSet(args.test_predict or args.test, 
                                                                                t._labels, legacy0=args.legacy0,
                                                                                keep_in_memory=True), 
                                            t._featureIndexes, cache_features=False)
        
        t.decode_dataset(evalData, print_predictions=(args.test_predict is not None), 
                         useBIO=args.bio, includeLossTerm=False, costAugVal=0.0)
        
    if args.predict is not None:
        # predict on a separate dataset
        
        predData = SupersenseFeaturizer(featureExtractor, SupersenseDataSet(args.predict, 
                                                                            t._labels, legacy0=args.legacy0, 
                                                                            keep_in_memory=False,
                                                                            autoreset=False),   # could be stdin, which should never be reset 
                                        t._featureIndexes, cache_features=False)

        t.decode_dataset(predData, print_predictions=True, useBIO=args.bio, includeLossTerm=False, costAugVal=0.0)
        
        
        
        

    
    elif args.test is None and args.weights:
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
