'''
Ported from Michael Heilman's SuperSenseFeatureExtractor.java
(parts refactored into the 'morph' module).
@author: Nathan Schneider (nschneid)
@since: 2012-07-22
'''
from __future__ import print_function, division
import sys, os, re, gzip

SRCDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = SRCDIR+'/../data'

_options = {'usePrefixAndSuffixFeatures': False, 
            'useClusterFeatures': False, 
            'useBigramFeatures': False, # token bigrams
            'WordNetPath': SRCDIR+'/../dict/file_properties.xml',
            "clusterFile": DATADIR+"/clusters/clusters_1024_49.gz",
            "useOldDataFormat": True}

def registerOpts(program_args):
    _options['usePrevLabel'] = not program_args.excludeFirstOrder

clusterMap = None

startSymbol = None
endSymbol = None


class Trie(object):
    '''
    Trie (prefix tree) data structure for mapping sequences to values.
    Values can be overridden, but removal from the trie is not currently supported.
    
    >>> t = Trie()
    >>> t['panther'] = 'PANTHER'
    >>> t['panda'] = 'PANDA'
    >>> t['pancake'] = 'PANCAKE'
    >>> t['pastrami'] = 'PASTRAMI'
    >>> t['pastafarian'] = 'PASTAFARIAN'
    >>> t['noodles'] = 'NOODLES'
        
    >>> for s in ['panther', 'panda', 'pancake', 'pastrami', 'pastafarian', 'noodles']:
    ...    assert s in t
    ...    assert t.get(s)==s.upper()
    >>> 'pescatarian' in t
    False
    >>> print(t.get('pescatarian'))
    None
    >>> t.longest('pasta', False)
    False
    >>> t.longest('pastafarian')
    (('p', 'a', 's', 't', 'a', 'f', 'a', 'r', 'i', 'a', 'n'), 'PASTAFARIAN')
    >>> t.longest('pastafarianism')
    (('p', 'a', 's', 't', 'a', 'f', 'a', 'r', 'i', 'a', 'n'), 'PASTAFARIAN')
    
    >>> t[(3, 1, 4)] = '314'
    >>> t[(3, 1, 4, 1, 5, 9)] = '314159'
    >>> t[(0, 0, 3, 1, 4)] = '00314'
    >>> t.longest((3, 1, 4))
    ((3, 1, 4), '314')
    >>> (3, 1, 4, 1, 5) in t
    False
    >>> print(t.get((3, 1, 4, 1, 5)))
    None
    >>> t.longest((3, 1, 4, 1, 5))
    ((3, 1, 4), '314')
    '''
    def __init__(self):
        self._map = {}  # map from sequence items to embedded Tries
        self._vals = {} # map from items ending a sequence to their values
    
    def __setitem__(self, seq, v):
        first, rest = seq[0], seq[1:]
        if rest:
            self._map.setdefault(first, Trie())[rest] = v
        else:
            self._vals[first] = v
    
    def __contains__(self, seq):
        '''@return: whether a value is stored for 'seq' '''
        first, rest = seq[0], seq[1:]
        if rest:
            if first not in self._map:
                return False
            return rest in self._map[first]
        return first in self._vals
    
    def get(self, seq, default=None):
        '''@return: value associated with 'seq' if 'seq' is in the trie, 'default' otherwise'''
        first, rest = seq[0], seq[1:]
        if rest:
            if first not in self._map:
                return default
            return self._map[first].get(rest)
        else:
            return self._vals.get(first, default)
        
    def longest(self, seq, default=None):
        '''@return: pair of longest prefix of 'seq' corresponding to a value in the Trie, and 
        that value. If no prefix of 'seq' has a value, returns 'default'.'''
        
        first, rest = seq[0], seq[1:]
        longer = self._map[first].longest(rest, default) if rest and first in self._map else default
        if longer==default: # 'rest' is empty, or none of the prefix of 'rest' leads to a value
            if first in self._vals:
                return ((first,), self._vals[first])
            else:
                return default
        else:
            return ((first,)+longer[0], longer[1])
        

senseTrie = None  # word1 -> {word2 -> ... -> {pos -> most frequent supersense}}
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
        self._len = len(self._i2s)  # cache the length for efficiency
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
    def __contains__(self, k):
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
        return self._len if self.is_frozen() else len(self._i2s)
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
    def __repr__(self):
        return 'IndexedFeatureMap(['+', '.join(map(repr,self.named_items()))+'])'



def extractFeatureValues(sent, j, usePredictedLabels=True, orders={0,1}, indexer=None):
    '''
    Extracts a map of feature names to values for a particular token in a sentence.
    These can be aggregated to get the feature vector or score for a whole sentence.
    These replicate the features used in Ciaramita and Altun, 2006 
    
    @param sent: the labeled sentence object to extract features from
    @param j: index of the word in the sentence to extract features for
    @param usePredictedLabels: whether to use predicted labels or gold labels (if available) 
    for the previous tag. This only applies to first-order features.
    @param orders: list of orders; e.g. if {1}, only first-order (tag bigram) features will be extracted
    @return: feature name -> value
    '''
    
    
    featureMap = IndexedFeatureMap(indexer) if indexer is not None else {}
    
    
    '''
        //sentence level lexicalized features
        //for(int i=0;i<sent.length();i++){
        //    if(j==i) continue;
        //    String sentStem = sent.getStems()[i);
        //    featureMap.put("curTok+sentStem="+curTok+"\t"+sentStem,1/sent.length());
        //}
    '''
    
    # note: in the interest of efficiency, we use tuples rather than string concatenation for feature names
    
    # previous label feature (first-order Markov dependency)
    if 1 in orders and hasFirstOrderFeatures() and j>0:
            featureMap["prevLabel=",(sent[j-1].prediction if usePredictedLabels else sent[j-1].gold)] = 1
    
    if 0 in orders:
        # bias
        featureMap["bias",] = 1
        
        # first sense features
        if sent.mostFrequentSenses is None or len(sent.mostFrequentSenses)!=len(sent):
            sent.mostFrequentSenses = extractFirstSensePredictedLabels(sent)
            
        firstSense = sent.mostFrequentSenses[j];
        
        if firstSense is None: firstSense = "0";
        featureMap["firstSense=",firstSense] = 1
        featureMap["firstSense+curTok=",firstSense,"\t",sent[j].stem] = 1
            
        useClusterFeatures = _options['useClusterFeatures']
        
        if useClusterFeatures:
            # cluster features for the current token
            curCluster = wordClusterID(sent[j].token)
            featureMap["firstSense+curCluster=",firstSense,"\t",curCluster] = 1
            featureMap["curCluster=",curCluster] = 1

            
        
        # word and POS features
        curPOS = sent[j].pos
        if curPOS=="NN" or curPOS=="NNS":
            featureMap["curPOS_common",] = 1
        if curPOS=="NNP" or curPOS=="NNPS":
            featureMap["curPOS_proper",] = 1
            
        featureMap["curTok=",sent[j].stem] = 1
        featureMap["curPOS=",curPOS] = 1
        featureMap["curPOS_0=",curPOS[0]] = 1
        
        
        useBigramFeatures = _options['useBigramFeatures']   # note: these are observation bigrams, not tag bigrams
        
        if j>0:
            featureMap["prevTok=",sent[j-1].stem] = 1
            if useBigramFeatures: featureMap["prevTok+curTok=",sent[j-1].stem,"\t",sent[j].stem] = 1
            featureMap["prevPOS=",sent[j-1].pos] = 1
            featureMap["prevPOS_0=",sent[j-1].pos[0]] = 1
            if useClusterFeatures: featureMap["prevCluster=",wordClusterID(sent[j-1].token)] = 1
            #if useClusterFeatures: featureMap["firstSense+prevCluster="+firstSense+"\t"+prevCluster] = 1
            
        if j+1<len(sent):
            featureMap["nextTok=",sent[j+1].stem] = 1
            if useBigramFeatures: featureMap["nextTok+curTok=",sent[j+1].stem,"\t",sent[j].stem] = 1
            featureMap["nextPOS=",sent[j+1].pos] = 1
            featureMap["nextPOS_0=",sent[j+1].pos[0]] = 1
            if useClusterFeatures: featureMap["nextCluster=",wordClusterID(sent[j+1].token)] = 1
            #if useClusterFeatures: featureMap["firstSense+nextCluster="+firstSense+"\t"+nextCluster] = 1
    
            
        if j-1>0:
            featureMap["prev2Tok=",sent[j-2].stem] = 1
            featureMap["prev2POS=",sent[j-2].pos] = 1
            featureMap["prev2POS_0=",sent[j-2].pos[0]] = 1
            if useClusterFeatures: featureMap["prev2Cluster=",wordClusterID(sent[j-2].token)] = 1
            #if useClusterFeatures: featureMap["firstSense+prev2Cluster="+firstSense+"\t"+prev2Cluster] = 1
            
        if j+2<len(sent):
            featureMap["next2Tok=",sent[j+2].stem] = 1
            featureMap["next2POS=",sent[j+2].pos] = 1
            featureMap["next2POS_0=",sent[j+2].pos[0]] = 1
            if useClusterFeatures: featureMap["next2Cluster=",wordClusterID(sent[j+2].token)] = 1
            #if useClusterFeatures: featureMap["firstSense+next2Cluster="+firstSense+"\t"+next2Cluster] = 1
            
            
        # word shape features
        
        featureMap["curShape=",sent[j].shape] = 1
            
        if j>0:
            featureMap["prevShape=",sent[j-1].shape] = 1
            
        if j+1<len(sent):
            featureMap["nextShape=",sent[j+1].shape] = 1
            
        if j-1>0:
            featureMap["prev2Shape=",sent[j-2].shape] = 1
            
        if j+2<len(sent):
            featureMap["next2Shape=",sent[j+2].shape] = 1
            
        firstCharCurTok = sent[j].token[0]
        if firstCharCurTok.lower()==firstCharCurTok:
            featureMap["curTokLowercase",] = 1
        elif j==0:
            featureMap["curTokUpperCaseFirstChar",] = 1
        else:
            featureMap["curTokUpperCaseOther",] = 1
        
        # 3-letter prefix and suffix features (disabled by default)
        if _options['usePrefixAndSuffixFeatures']:
            featureMap["prefix=",sent[j].token[:3]] = 1
            featureMap["suffix=",sent[j].token[-3:]] = 1
    
    
    return featureMap



def extractFirstSensePredictedLabels(sent):
    '''
    Extract most frequent sense baseline from WordNet data,
    using Ciaramita and Altun's approach.  Also, uses
    the data from the Supersense tagger release.
    
    @param sent: labeled sentence
    @return list of predicted labels
    '''
    
    if not senseTrie:
        if _options['useOldDataFormat']:
            loadSenseDataOriginalFormat()
        else:
            loadSenseDataNewFormat()

    res = []
    
    prefix = "B-"
    phrase = None
    
    stems = [tok.stem for tok in sent]
    poses = [tok.pos for tok in sent]
    for i in range(len(sent)):
        mostFrequentSense = None
        
        pos = sent[i].pos
        '''
        for j in range(len(sent)-1, i-1, -1):
            #phrase = '_'.join(stems[i:j+1])    # SLOW
            wordParts = tuple(stems[i:j+1])
            endPos = sent[j].pos
            mostFrequentSense = getMostFrequentSense(wordParts, pos[:1])
            if mostFrequentSense is not None: break
            mostFrequentSense = getMostFrequentSense(wordParts, endPos[:1])
            if mostFrequentSense is not None: break
        '''
        
        mostFrequentSense = getMostFrequentSensePrefix(stems[i:], poses[i:])
        if mostFrequentSense:
            wordParts, mostFrequentSense = mostFrequentSense
            res.append(intern('B-'+mostFrequentSense))
            i += 1
            for word in wordParts[1:]:
                res.append(intern('I-'+mostFrequentSense))
                i += 1
        else:
            res.append("0")
        
    return res

def getMostFrequentSensePrefix(stems, poses):
    '''
    Look up the most frequent sense of the words in a phrase and 
    their POSes.
    '''
    if not senseTrie:
        assert _options['useOldDataFormat']
        loadSenseDataOriginalFormat()
        
    pos2sense = senseTrie.longest(stems)
    if pos2sense:
        prefix, pos2sense = pos2sense
        if poses[0] in pos2sense:
            return pos2sense[poses[0]]
        elif poses[-1] in pos2sense:
            return pos2sense[poses[-1]]
        
        # neither the first nor last POS of the longest-matching 
        # series of words was a match. try a shorter prefix (rare).
        stems, poses = stems[:len(prefix)-1], poses[:len(prefix)-1]
        if stems:
            return getMostFrequentSensePrefix(stems, poses)
    return None

def loadSenseDataNewFormat():
    '''
    Load morphology and sense information provided by the 
    Supersense tagger release from SourceForge.
    '''
    print("loading most frequent sense information...", file=sys.stderr)
    
    global possibleSensesMap, senseCountMap
    assert not possibleSensesMap
    assert not senseCountMap
    
    nounFile = _options.setdefault("nounFile",DATADIR+"/oldgaz/NOUNS_WS_SS_P.gz")
    verbFile = _options.setdefault("verbFile",DATADIR+"/oldgaz/VERBS_WS_SS.gz")
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
        possibleSensesFile = _options.setdefault("possibleSensesFile",DATADIR+"/gaz/possibleSuperSenses.GAZ.gz")
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
    global possibleSensesMap, senseTrie, senseCountMap
    assert not senseTrie
    senseTrie = Trie()
    assert not possibleSensesMap
    assert not senseCountMap
    
    nounFile = _options.setdefault("nounFile",DATADIR+"/oldgaz/NOUNS_WS_SS_P.gz")
    _loadSenseFileOriginalFormat(nounFile, "N")
    verbFile = _options.setdefault("verbFile",DATADIR+"/oldgaz/VERBS_WS_SS.gz")
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
    if phrase not in senseTrie:
        senseTrie[phrase] = {simplePOS: intern(sense)}
    else:
        senseTrie.get(phrase)[simplePOS] = intern(sense)
    
    senseCountMap.setdefault(simplePOS, {})[phrase] = numSenses


if __name__=='__main__':
    import doctest
    doctest.testmod()
