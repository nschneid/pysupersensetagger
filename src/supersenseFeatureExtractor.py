'''
Ported from Michael Heilman's SuperSenseFeatureExtractor.java
(parts refactored into the 'morph' module).
@author: Nathan Schneider (nschneid)
@since: 2012-07-22
'''
from __future__ import print_function, division
import sys, re, gzip

_options = {'usePrefixAndSuffixFeatures': False, 
            'useClusterFeatures': False, 
            'useBigramFeatures': False, # token bigrams
            'usePrevLabel': True,   # label bigrams (first-order)
            'WordNetPath': 'dict/file_properties.xml',
            "clusterFile": "../data/clusters/clusters_1024_49.gz",
            "useOldDataFormat": True}

clusterMap = None

startSymbol = None
endSymbol = None

senseMap = {} # pos -> {word -> most frequent supersense}
senseCountMap = {} # pos -> {word -> number of senses}
possibleSensesMap = {} # stem -> set of possible supersenses


def loadDefaults():
    # TODO: properties allowing for override of _options defaults
    global clusterMap
    if _options['useClusterFeatures'] and clusterMap is None:
        # load clusters
        print("loading word cluster information...", file=sys.stderr);
        clusterMap = {}
        clusterFile = _options['clusterFile']
        with gzip.open(clusterFile) as clusterF:
            clusterID = 0
            for ln in clusterF:
                parts = re.split(r'\\s', ln)
                for part in parts:
                    clusterMap[part] = clusterID
                clusterID += 1
        print("done.", file=sys.stderr);
        

def hasFirstOrderFeatures():
    return _options['usePrevLabel']

def wordClusterID(word):
    cid = clusterMap.get(word.lower(), 'UNK')
    if cid=='UNK': return cid
    return 'C'+str(cid)

class SequentialStringIndexer(object):
    def __init__(self):
        self._s2i = {}
        self._i2s = []
        self._frozen = False
    def freeze(self):
        self._frozen = True
    def unfreeze(self):
        self._frozen = False
    def is_frozen(self):
        return self._frozen
    def __getitem__(self, k):
        if isinstance(k,int):
            return self._i2s[k]
        return self._s2i[k]
    def get(self, k, default=None):
        if isinstance(k,int):
            if k>=len(self._i2s):
                return default
            return self._i2s[k]
        return self._s2i.get(k,default)
    def __in__(self, k):
        if isinstance(k,int):
            assert k>0
            return k<len(self._i2s)
        return k in self._s2i
    def add(self, s):
        if s not in self:
            if self.is_frozen():
                raise ValueError('Cannot add new item to frozen indexer: '+s)
            self._s2i[s] = len(self._i2s)
            self._i2s.append(s)
    def setdefault(self, k):
        '''looks up k, adding it if necessary'''
        self.add(k)
        return self[k]
    def __len__(self):
        return len(self._i2s)
    @property
    def strings(self):
        return self._i2s
    def items(self):
        '''iterator over (index, string) pairs, sorted by index'''
        return enumerate(self._i2s)

class IndexedStringSet(set):
    '''Wraps a set contains indices to strings, with mapping provided in an indexer.'''
    def __init__(self, indexer):
        self._indexer = indexer
        self._indices = set()
    @property
    def strings(self):
        return {self._indexer[i] for i in self._indices}
    @property
    def indices(self):
        return self._indices
    def __iter__(self):
        return iter(self._indices)
    def __len__(self):
        return len(self._indices)
    def add(self, k):
        '''If k is not already indexed, index it before adding it to the set'''
        if isinstance(k, int):
            assert k in self._indexer
            self._indices.add(k)
        else:
            self._indices.add(self._indexer.setdefault(k))
    def setdefault(self, k):
        self.add(k)
        return self._indexer[k]

class IndexedFeatureMap(object):
    def __init__(self, indexer, default=1):
        self._set = IndexedStringSet(indexer)
        self._map = {}
        self._default = default
    def __setitem__(self, k, v):
        '''Add the specified feature/value pair if the feature is already indexed or 
        can be added to the index. Has no effect if the featureset is frozen and the 
        provided feature is not part of the featureset.'''
        if not self._set._indexer.is_frozen() or k in self._set._indexer:
            i = self._set.setdefault(k)
            if v!=self._default:
                self._map[i] = v
    def __iter__(self):
        return iter(self._set)
    def __len__(self):
        return len(self._set)
    def items(self):
        for i in self._set:
            yield (i, self._map.get(i, self._default))
    def named_items(self):
        for i in self._set:
            yield (self._set._indexer[i], self._map.get(i, self._default))

def extractFeatureValues(sent, j, usePredictedLabels=True, orders={0,1}, indexer=None):
    '''
    Extracts a map of feature names to values for a particular token in a sentence.
    These can be aggregated to get the feature vector or score for a whole sentence.
    These replicate the features used in Ciaramita and Altun, 2006 
    
    @param sent: the labeled sentence object to extract features from
    @param j: index of the word in the sentence to extract features for
    @param usePredictedLabels: whether to use predicted labels or gold labels (if available)
    @param orders: list of orders; e.g. if {1}, only first-order (tag bigram) features will be extracted
    @return: feature name -> value
    '''
    
    featureMap = IndexedFeatureMap(indexer) if indexer is not None else {}
    
    _orders = orders
    if 1 in _orders and not hasFirstOrderFeatures():
        _orders = set(_orders)
        _orders.remove(1)
    
    useClusterFeatures = _options['useClusterFeatures']
    useBigramFeatures = _options['useBigramFeatures']   # note: these are observation bigrams, not tag bigrams
    usePrefixAndSuffixFeatures = _options['usePrefixAndSuffixFeatures']
    
    curTok, curStem, curPOS, _, _, curShape = sent[j]
    curCluster = wordClusterID(curTok) if useClusterFeatures else None
    
    prevLabel = startSymbol
            
    if sent.mostFrequentSenses is None or len(sent.mostFrequentSenses)!=len(sent):
        sent.mostFrequentSenses = extractFirstSensePredictedLabels(sent)
        
    firstSense = sent.mostFrequentSenses[j];
    
    prevShape = startSymbol
    prevPOS = startSymbol
    prevStem = startSymbol
    prevCluster = startSymbol
    nextShape = startSymbol
    nextPOS = endSymbol
    nextStem = endSymbol
    nextCluster = endSymbol
        
    prev2Shape = startSymbol
    prev2POS = startSymbol
    prev2Stem = startSymbol
    prev2Cluster = startSymbol
    next2Shape = startSymbol
    next2POS = endSymbol
    next2Stem = endSymbol
    next2Cluster = endSymbol
    
    if j-2 >= 0:
        prev2Shape = sent[j-2].shape
        prev2Stem = sent[j-2].stem
        prev2POS = sent[j-2].pos
        if useClusterFeatures: prev2Cluster = wordClusterID(sent[j-2].token)
        
    if j-1 >= 0:
        prevTok, prevStem, prevPOS, prevGold, prevPred, prevShape = sent[j-1]
        if useClusterFeatures: prevCluster = wordClusterID(prevTok)
        if usePredictedLabels:
            prevLabel = prevPred
        else:
            prevLabel = prevGold
            
        
    if j+1 < len(sent):
        nextShape = sent[j+1].shape
        nextStem = sent[j+1].stem
        nextPOS = sent[j+1].pos
        if useClusterFeatures: nextCluster = wordClusterID(sent[j+1].token)
        
    if j+2 < len(sent):
        next2Shape = sent[j+2].shape
        next2Stem = sent[j+2].stem
        next2POS = sent[j+2].pos
        if useClusterFeatures: next2Cluster = wordClusterID(sent[j+2].token);
        
    '''
        //sentence level lexicalized features
        //for(int i=0;i<sent.length();i++){
        //    if(j==i) continue;
        //    String sentStem = sent.getStems()[i);
        //    featureMap.put("curTok+sentStem="+curTok+"\t"+sentStem,1/sent.length());
        //}
    '''
    
    # note: in string concatenations below, we use ''.join([x,y]) instead of x+y for efficiency reasons
    
    if 0 in _orders:
        # bias
        featureMap["bias",] = 1
            
        # first sense features
        if firstSense is None: firstSense = "0";
        featureMap["firstSense=",firstSense] = 1
        featureMap["firstSense+curTok=",firstSense,"\t",curStem] = 1
            
            
        if useClusterFeatures:
            # cluster features for the current token
            featureMap["firstSense+curCluster=",firstSense,"\t",curCluster] = 1
            featureMap["curCluster=",curCluster] = 1

        
        
    # previous label feature (first-order Markov dependency)
    if 1 in _orders and prevLabel!=startSymbol: featureMap["prevLabel=",prevLabel] = 1
        
    if 0 in _orders:
        # word and POS features
        if curPOS=="NN" or curPOS=="NNS":
            featureMap["curPOS_common",] = 1
        if curPOS=="NNP" or curPOS=="NNPS":
            featureMap["curPOS_proper",] = 1
            
        featureMap["curTok=",curStem] = 1
        featureMap["curPOS=",curPOS] = 1
        featureMap["curPOS_0=",curPOS[0]] = 1
        
        if prevPOS != startSymbol:
            featureMap["prevTok=",prevStem] = 1
            if useBigramFeatures: featureMap["prevTok+curTok=",prevStem,"\t",curStem] = 1
            featureMap["prevPOS=",prevPOS] = 1
            featureMap["prevPOS_0=",prevPOS[0]] = 1
            if useClusterFeatures: featureMap["prevCluster=",prevCluster] = 1
            #if useClusterFeatures: featureMap["firstSense+prevCluster="+firstSense+"\t"+prevCluster] = 1
            
        if nextPOS != endSymbol:
            featureMap["nextTok=",nextStem] = 1
            if useBigramFeatures: featureMap["nextTok+curTok=",nextStem,"\t",curStem] = 1
            featureMap["nextPOS=",nextPOS] = 1
            featureMap["nextPOS_0=",nextPOS[0]] = 1
            if useClusterFeatures: featureMap["nextCluster=",nextCluster] = 1
            #if useClusterFeatures: featureMap["firstSense+nextCluster="+firstSense+"\t"+nextCluster] = 1
    
            
        if prev2POS != startSymbol:
            featureMap["prev2Tok=",prev2Stem] = 1
            featureMap["prev2POS=",prev2POS] = 1
            featureMap["prev2POS_0=",prev2POS[0]] = 1
            if useClusterFeatures: featureMap["prev2Cluster=",prev2Cluster] = 1
            #if useClusterFeatures: featureMap["firstSense+prev2Cluster="+firstSense+"\t"+prev2Cluster] = 1
            
        if next2POS != endSymbol:
            featureMap["next2Tok=",next2Stem] = 1
            featureMap["next2POS=",next2POS] = 1
            featureMap["next2POS_0=",next2POS[0]] = 1
            if useClusterFeatures: featureMap["next2Cluster=",next2Cluster] = 1
            #if useClusterFeatures: featureMap["firstSense+next2Cluster="+firstSense+"\t"+next2Cluster] = 1
            
            
        # word shape features
        
        featureMap["curShape=",curShape] = 1
            
        if prevPOS != startSymbol:
            featureMap["prevShape=",prevShape] = 1
            
        if nextPOS != endSymbol:
            featureMap["nextShape=",nextShape] = 1
            
        if prev2POS != startSymbol:
            featureMap["prev2Shape=",prev2Shape] = 1
            
        if next2POS != endSymbol:
            featureMap["next2Shape=",next2Shape] = 1
            
        firstCharCurTok = curTok[0]
        if firstCharCurTok.lower()==firstCharCurTok:
            featureMap["curTokLowercase",] = 1
        elif j==0:
            featureMap["curTokUpperCaseFirstChar",] = 1
        else:
            featureMap["curTokUpperCaseOther",] = 1
        
        # 3-letter prefix and suffix features (disabled by default)
        if usePrefixAndSuffixFeatures:
            featureMap["prefix=",curTok[:3]] = 1
            featureMap["suffix=",curTok[-3:]] = 1
    
    
    return featureMap


def extractFirstSensePredictedLabels(sent):
    '''
    Extract most frequent sense baseline from WordNet data,
    using Ciaramita and Altun's approach.  Also, uses
    the data from the Supersense tagger release.
    
    @param sent: labeled sentence
    @return list of predicted labels
    '''
    
    if not senseMap:
        if _options['useOldDataFormat']:
            loadSenseDataOriginalFormat()
        else:
            loadSenseDataNewFormat()

    res = []
    
    prefix = "B-"
    phrase = None
    
    stems = [tok.stem for tok in sent]
    for i in range(len(sent)):
        mostFrequentSense = None
        
        pos = sent[i].pos
        for j in range(len(sent)-1, i-1, -1):
            #phrase = '_'.join(stems[i:j+1])    # SLOW
            wordParts = tuple(stems[i:j+1])
            endPos = sent[j].pos
            mostFrequentSense = getMostFrequentSense(wordParts, pos[:1])
            if mostFrequentSense is not None: break
            mostFrequentSense = getMostFrequentSense(wordParts, endPos[:1])
            if mostFrequentSense is not None: break
        
        prefix = "B-";
        if mostFrequentSense is not None:
            while i<j:
                res.append(intern(prefix+mostFrequentSense))
                prefix = "I-"
                i += 1
            
        if mostFrequentSense is not None:
            res.append(intern(prefix+mostFrequentSense))
        else:
            res.append("0")
        
    return res

def getMostFrequentSense(phrase, pos):
    '''
    Look up the most frequent sense of an underscore-separated 
    phrase and its POS.
    '''
    if not senseMap:
        if _options['useOldDataFormat']:
            loadSenseDataOriginalFormat()
        else:
            loadSenseDataNewFormat()
    
    return senseMap[pos].get(phrase) if pos in senseMap else None

def loadSenseDataNewFormat():
    '''
    Load morphology and sense information provided by the 
    Supersense tagger release from SourceForge.
    '''
    print("loading most frequent sense information...", file=sys.stderr)
    
    global possibleSensesMap, senseMap, senseCountMap
    assert not possibleSensesMap
    assert not senseMap
    assert not senseCountMap
    
    nounFile = _options.setdefault("nounFile","../data/oldgaz/NOUNS_WS_SS_P.gz")
    verbFile = _options.setdefault("verbFile","../data/oldgaz/VERBS_WS_SS.gz")
    for pos,f in [('N',nounFile), ('V',verbFile)]:
        with gzip.open(f) as inF:
            for ln in inF:
                parts = ln[:-1].split('\t')
                try:
                    sense = parts[1][parts[1].indexOf('=')+1:]
                    numSenses = int(parts[3][parts[3].indexOf('=')+1:])
                    _addMostFrequentSense(parts[0], pos, sense, numSenses)
                except IndexError:
                    print(parts)
                    raise
    
    try:
        possibleSensesFile = _options.setdefault("possibleSensesFile","../data/gaz/possibleSuperSenses.GAZ.gz")
        with gzip.open(possibleSensesFile) as psF:
            for ln in psF:
                parts = ln[:-1].split('\t')
                wordParts = parts[0].split('_')
                for j in range(len(wordParts)):
                    tmp = possibleSensesMap.get(wordParts[j], set())
                    for i in range(1,len(parts)):
                        tmp.add(('B-' if j==0 else 'I-')+parts[i])                
                    possibleSensesMap[wordParts[j]] = tmp
    except IOError as ex:
        print(ex.message, file=sys.stderr)
    
    print("done.", file=sys.stderr)

def loadSenseDataOriginalFormat():
    '''
    Load data from the original SST release
    '''
    print("loading most frequent sense information (old format)...", file=sys.stderr)
    global possibleSensesMap, senseMap, senseCountMap
    assert not possibleSensesMap
    assert not senseMap
    assert not senseCountMap
    
    nounFile = _options.setdefault("nounFile","../data/oldgaz/NOUNS_WS_SS_P.gz")
    _loadSenseFileOriginalFormat(nounFile, "N")
    verbFile = _options.setdefault("verbFile","../data/oldgaz/VERBS_WS_SS.gz")
    _loadSenseFileOriginalFormat(verbFile, "V")

    print("done.", file=sys.stderr)

def _loadSenseFileOriginalFormat(senseFile, shortPOS):
    spacing = 2 if shortPOS=='V' else 3 # the noun file has 3 sets of columns, the verb file has 2
    with gzip.open(senseFile) as senseF:
        for ln in senseF:
            ln = ln[:-1]
            if not ln: continue
            parts = re.split(r'\s', ln)
            sense = parts[2]
            numSenses = (len(parts)-1)//spacing

            # multiwords
            wordParts = tuple(parts[0].split("_"))

            # first sense listed is the most frequent one
            # record it for the most frequent sense baseline algorithm
            _addMostFrequentSense(wordParts, shortPOS, sense, numSenses)
            # store tuple of words instead of underscore-separated lemma, for efficiency reasons on lookup
            
            # read possible senses, split up multi word phrases
            
            for i in range(2,len(parts),spacing):
                for j in range(len(wordParts)):
                    tmp = possibleSensesMap.get(wordParts[j], set())
                    possibleSense = parts[i]
                    tmp.add(('B-' if j==0 else 'I-')+possibleSense) # TODO: interning?
                    possibleSensesMap[wordParts[j]] = tmp

def _addMostFrequentSense(phrase, simplePOS, sense, numSenses):
    '''
    Store the most frequent sense and its count.
    '''
    senseMap.setdefault(simplePOS, {})[phrase] = intern(sense)  # TODO: more interning?
    senseCountMap.setdefault(simplePOS, {})[phrase] = numSenses
