'''
Ported from Michael Heilman's SuperSenseFeatureExtractor.java
(parts refactored into the 'morph' module).
@author: Nathan Schneider (nschneid)
@since: 2012-07-22
'''
from __future__ import print_function, division
import sys, os, re, gzip, codecs, json
from collections import Counter, defaultdict

SRCDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = SRCDIR+'/../data'

_options = {'usePrefixAndSuffixFeatures': False, 
            'useClusterFeatures': False, 
            'useClusterPrefixFeatures': False,
            'useBigramFeatures': False, # token bigrams
            'useFirstSensePlusToken': False,
            'useContextPOSFilter': False,
            'WordNetPath': SRCDIR+'/../dict/file_properties.xml',
            "clusterFile": DATADIR+"/clusters/clusters_1024_49.gz",
            "useOldDataFormat": True,
            'usePOSNeighborFeatures': False}

def registerOpts(program_args):
    _options['usePrevLabel'] = not program_args.excludeFirstOrder
    _options['useBigramFeatures'] = program_args.bigrams
    _options['useClusterFeatures'] = program_args.clusters
    _options['clusterFile'] = program_args.cluster_file
    _options['usePOSNeighborFeatures'] = program_args.pos_neighbors
    _options['useContextPOSFilter'] = program_args.cxt_pos_filter
    
    loadDefaults()
    if program_args.lex is not None:
        loadLexicons(program_args.lex)

clusterMap = None
lexicons = defaultdict(list)

startSymbol = None
endSymbol = None


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


def loadDefaults(oldClusterFormat=False):
    # TODO: properties allowing for override of _options defaults
    global clusterMap
    if _options['useClusterFeatures'] and clusterMap is None:
        # load clusters
        print("loading word cluster information...", file=sys.stderr);
        clusterMap = {}
        clusterFile = _options['clusterFile']
        with gzip.open(clusterFile) as clusterF:
            if oldClusterFormat:    # each line is a cluster, with space-separated words
                clusterID = 0
                for ln in clusterF:
                    ww = re.split(r'\s', ln)
                    for w in ww:
                        clusterMap[w] = clusterID
                    clusterID += 1
            else:   # each line contains a cluster ID (bitstring for Brown clusters), a word type, and a count
                for ln in clusterF:
                    clusterID, w, n = ln[:-1].split('\t')
                    clusterMap[w.decode('utf-8')] = clusterID
                    
                
        print("done.", file=sys.stderr);
        

def loadLexicons(lexfiles):
    global lexicons
    # index each entry by longest word
    for lexfile in lexfiles:
        for ln in lexfile:
            entry = json.loads(ln[:-1].decode('utf-8'))
            entry_words = entry["lemmas"] if entry.get("lemmas") else entry["words"]
            if not any(w for w in entry_words if len(w)>2):
                continue    # probably garbage entry
            if entry_words[-1]=='the':
                continue    # probably garbage entry
            key = sorted(entry_words, key=lambda w: (len(w),w))[0]
            lexicons[key].append(entry)

def extractLexiconCandidates(sent, lowercase=True):
    sent_cands = []
    contig = defaultdict(list)
    gappy = []
    tokmap = defaultdict(set)
    for j,t in enumerate(sent):
        tokmap[t.token].add(j)
    toks = set(tokmap.keys())
    sent_tokens = [t.token.lower() if lowercase else t.token for t in sent]
    sent_stems = [t.stem.lower() if lowercase else t.stem for t in sent]
    for tok in toks:
        if tok in lexicons:
            for entry in lexicons[tok]:
                if "lemmas" in entry:
                    entry_words = [w.lower() if lowercase else w for w in entry["lemmas"]]
                    entry["lemmas"] = entry_words
                    sent_words = sent_stems
                else:
                    entry_words = [w.lower() if lowercase else w for w in entry["words"]]
                    entry["words"] = entry_words
                    sent_words = sent_tokens
                
                if set(entry_words)<=toks:  # all entry words are in the sentence
                    if entry not in sent_cands:
                        sent_cands.append(entry)
                    entryLen = len(entry_words)
                    isContig = False  # can this occurrence be contiguous?
                    for j in tokmap[tok]:
                        for i in range(max(j-entryLen,0),min(j+entryLen,len(sent))):
                            # all matched-length spans of the sentence including j
                            if set(sent_words[j-i:j-i+entryLen])==set(entry_words):
                                # found a contiguous occurrence! not necessarily matching the order of words in the entry
                                isContig = True
                                for k in range(i,i+entryLen):
                                    contig[k].append((i, entry))
                                    
                    if not isContig and entry not in gappy:
                        if not any(len(w)>3 for w in entry_words):
                            continue    # probably all function words, not intended to be gappy
                        if len(entry_words)==2 and ('the' in entry_words or 'a' in entry_words or 'an' in entry_words or 'of' in entry_words):
                            continue    # probably not really gappy
                        longestLen = max(len(w) for w in entry_words)
                        ok = False
                        for w in entry_words:
                            if len(w)<longestLen: continue
                            if any(t for it,t in enumerate(sent_words) if t==w and sent[it].pos not in {'DT','PDT','IN','MD','CC','PRP','PRP$','WDT','WRB','WP','WP$'}):
                                ok = True
                        if not ok:  # longest words are function words. probably should not be gappy
                            continue
                        
                        # construct a regex of the entry words & the sentence to see if order is preserved
                        entryR = ' ' + r' .* '.join(re.escape(w) for w in entry_words) + ' '
                        if not re.search(entryR, ' '+' '.join(sent_words)+' ', re.U):
                            continue    # ordering mismatch
                        
                        gappy.append(entry)
    
    return contig, gappy

def hasFirstOrderFeatures():
    return _options['usePrevLabel']

def wordClusterID(word):
    cid = clusterMap.get(word.lower(), 'UNK')
    if cid=='UNK': return cid
    return 'C'+str(cid)



class SequentialStringIndexer(object):
    '''Feature alphabet. Optional cutoff threshold; count is determined by the number of 
    calls to add() or setdefault() before calling freeze().'''
    def __init__(self, cutoff=None):
        self._s2i = {}
        self._i2s = []
        self._frozen = False
        self._cutoff = cutoff
        if self._cutoff is not None:
            self._counts = Counter()
    def setcount(self, k, n):
        '''Store a count for the associated entry. Useful for entries that should be 
        kept when thresholding but which might not be counted by calls to add()/setdefault().'''
        i = k if isinstance(k,int) else self._s2i[k]
        self._counts[i] = n
    def freeze(self):
        '''Locks the alphabet so it can no longer be modified and filters 
        according to the cutoff threshold (if specified).'''
        if self._cutoff is not None:  # apply feature cutoff (threshold)
            assert self._counts
            self._i2s = [s for i,s in enumerate(self._i2s) if self._counts[i]>=self._cutoff]
            if 0<=self._cutoff<=1: assert len(self._i2s)==len(self._s2i)
            del self._counts
            self._s2i = {s: i for i,s in enumerate(self._i2s)}
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
            self._s2i[s] = i = len(self._i2s)
            self._i2s.append(s)
        elif not self.is_frozen() and self._cutoff is not None:
            i = self[s]
        if not self.is_frozen() and self._cutoff is not None:
            # update count
            self._counts[i] += 1
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
    '''Wraps a set contains indices to strings, with mapping provided in an indexer.
    Used to hold the features active for a particular instance.
    '''
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
    '''The feature-value mapping for a particular instance.'''
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

@memoize
def coarsen(pos):
    
    if pos=='TO': return 'I'
    elif pos.startswith('NNP'): return '^'
    elif pos=='CC': return '&'
    elif pos=='CD': return '#'
    elif pos=='RP': return 'T'
    else: return pos[0]

CPOS_PAIRS = [{'V','V'},{'V','N'},{'V','R'},{'V','T'},{'V','M'},{'V','P'},
              {'J','N'},{'N','N'},{'D','N'},{'D','^'},{'N','^'},{'^','^'},
              {'R','J'},{'N','&'},{'^','&'},{'V','I'},{'I','N'}]

DIGIT_RE = re.compile(r'\d')

def extractFeatureValues(sent, j, usePredictedLabels=True, orders={0,1}, indexer=None,
                         candidatesThisSentence=None):
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
            assert len(sent)==len(sent.mostFrequentSenses)
            
        firstSense = sent.mostFrequentSenses[j]
        
        if firstSense is None: firstSense = 'O'
        featureMap["firstSense=",firstSense] = 1
        if _options['useFirstSensePlusToken']:
            featureMap["firstSense+curTok=",firstSense,sent[j].stem] = 1
        
        useClusterFeatures = _options['useClusterFeatures']
        
        '''
        if useClusterFeatures:
            # cluster features for the current token
            curCluster = wordClusterID(sent[j].token)
            featureMap["firstSense+curCluster=",firstSense,curCluster] = 1
            featureMap["curCluster=",curCluster] = 1
        '''
         
        
        useBigramFeatures = _options['useBigramFeatures']   # note: these are observation bigrams, not tag bigrams
        
        if useBigramFeatures:
            if j>0:
                if not _options['useContextPOSFilter'] or {coarsen(sent[j-1].pos),coarsen(sent[j].pos)} in CPOS_PAIRS:
                    featureMap["prevStem+curStem=",sent[j-1].stem,"\t",sent[j].stem] = 1
            if j+1<len(sent):
                if not _options['useContextPOSFilter'] or {coarsen(sent[j+1].pos),coarsen(sent[j].pos)} in CPOS_PAIRS:
                    featureMap["nextStem+curStem=",sent[j+1].stem,"\t",sent[j].stem] = 1
        
        if useClusterFeatures: clusterj = wordClusterID(sent[j].token)
        for k in range(max(0,j-2),min(len(sent),j+3)):
            delta = '@{}='.format(k-j) if k!=j else ''
            featureMap["stem"+delta,sent[k].stem] = 1
            featureMap["pos"+delta,sent[k].pos] = 1
            featureMap["cpos"+delta,coarsen(sent[k].pos)] = 1
            featureMap["stem@0+pos"+delta,sent[j].stem,sent[k].pos] = 1
            featureMap["stem@0+cpos"+delta,sent[j].stem,coarsen(sent[k].pos)] = 1
            if useClusterFeatures:
                cluster = wordClusterID(sent[k].token)
                featureMap["cluster"+delta,cluster] = 1
                if k!=j: featureMap["cluster6+cluster6"+delta,clusterj[:6],cluster[:6]] = 1
                if _options['useClusterPrefixFeatures'] and cluster!='UNK':
                    for prefixlen in range(3,max(len(cluster),len(clusterj)),2): # even-length prefixes of the bitstring (cluster ID stats with "C")
                        featureMap["cluster"+delta,'c'+cluster[1:prefixlen]] = 1
                        if k!=j and clusterj!='UNK': featureMap["cluster+cluster"+delta,'c'+clusterj[1:prefixlen],'c'+cluster[1:prefixlen]] = 1
            #if useClusterFeatures: featureMap["firstSense+prevCluster="+firstSense+"\t"+prevCluster] = 1
            featureMap["shape"+delta,sent[k].shape] = 1
        
        
        sentpos = ''.join(coarsen(w.pos) for w in sent)
        cposj = coarsen(sent[j].pos)
        
        if candidatesThisSentence is not None:
            contig, gappy = candidatesThisSentence
            nContig = 0
            for c,entry in contig[j]:
                lbl = entry["label"] if not DIGIT_RE.search(entry["label"]) else ''
                #print(entry,file=sys.stderr)
                # TODO: check for ordering match against the lexical entry ordering?
                featureMap["contigMatch,first="+('1' if c==j else '0'),entry["datasource"],lbl] = 1
                for z in ("count","pmi","t","mle","dice"):
                    if z in entry:
                        featureMap["contigMatch,first="+('1' if c==j else '0'),entry["datasource"],lbl,z] = entry[z]
                featureMap["contigMatch",entry["datasource"],lbl,"poses=",sentpos[c:c+len(entry["lemmas"] if entry.get("lemmas") else entry["words"])],'@',str(j-c)] = 1
                # TODO: look at label?
                nContig += 1
            for n in range(1,nContig+1):
                featureMap["contigMatches>=",str(n)] = 1
            
            nGappy = 0
            for entry in gappy:
                lbl = entry["label"] if not DIGIT_RE.search(entry["label"]) else ''
                #print('GAPPY',entry,file=sys.stderr)
                # TODO: constrain number & placement of gaps to consider this a match?
                if (entry.get("lemmas") and sent[j].stem in entry["lemmas"]) or (entry.get("words") and sent[j].token in entry["words"]):
                    featureMap["gappyMatch",entry["datasource"],lbl] = 1
                    for z in ("count","pmi","t","mle","dice"):
                        if z in entry:
                            featureMap["gappyMatch",entry["datasource"],lbl,z] = entry[z]
                    if "poses" in entry and entry["poses"]:
                        featureMap["gappyMatch",entry["datasource"],lbl,"entryposes=",' '.join(entry["poses"]),"cpos=",cposj] = 1
                    nGappy += 1
            for n in range(1,nGappy+1):
                featureMap["gappyMatches>=",str(n)] = 1
        
        
        if _options['usePOSNeighborFeatures']:    # new feature
            for cpos in 'VN^ITPJRDM#&':
                if {cpos,cposj} not in CPOS_PAIRS:
                    continue
                if cpos in sentpos[:j]:
                    k = sentpos.rindex(cpos,0,j)
                    bindist = k-j
                    if abs(bindist)>5:
                        if abs(bindist)<10:
                            bindist = 6 if bindist>0 else -6
                        else:
                            bindist //= 10
                            bindist *= 10
                    featureMap[cpos,'<-{}'.format(bindist),cposj] = 1
                    featureMap[cpos,'<',sent[k].stem,cposj] = 1
                    if _options['useBigramFeatures'] and abs(bindist)<6:
                        featureMap[cpos,'<-{}'.format(bindist),sent[j].stem] = 1
                        featureMap[cpos,'<',sent[k].stem,sent[j].stem] = 1
                if cpos in sentpos[j+1:]:
                    k = sentpos.index(cpos,j+1)
                    bindist = k-j
                    if abs(bindist)>5:
                        if abs(bindist)<10:
                            bindist = 6 if bindist>0 else -6
                        else:
                            bindist //= 10
                            bindist *= 10
                    featureMap[cpos,'{}->'.format(bindist),cposj] = 1
                    featureMap[cpos,'>',sent[k].stem,cposj] = 1
                    if _options['useBigramFeatures'] and abs(bindist)<6:
                        featureMap[cpos,'{}->'.format(bindist),sent[j].stem] = 1
                        featureMap[cpos,'>',sent[k].stem,sent[j].stem] = 1
        
        
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
    coarse_poses = [tok.pos[0] for tok in sent]
    i = 0
    while i < len(sent):
        mostFrequentSenseResult = None
        
        #pos = sent[i].pos
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
        
        mostFrequentSenseResult = getMostFrequentSensePrefix(stems[i:], coarse_poses[i:])
        if mostFrequentSenseResult:
            wordParts, mostFrequentSense = mostFrequentSenseResult
            res.append(intern('B-'+mostFrequentSense))
            i += 1
            for word in wordParts[1:]:
                res.append(intern('I-'+mostFrequentSense))
                i += 1
        else:
            res.append('O')
            i += 1
    
    return res

def getMostFrequentSensePrefix(stems, coarse_poses):
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
        if coarse_poses[0] in pos2sense:
            return prefix, pos2sense[coarse_poses[0]]
        elif coarse_poses[-1] in pos2sense:
            return prefix, pos2sense[coarse_poses[-1]]
        
        # neither the first nor last POS of the longest-matching 
        # series of words was a match. try a shorter prefix (rare).
        stems, coarse_poses = stems[:len(prefix)-1], coarse_poses[:len(prefix)-1]
        if stems:
            return getMostFrequentSensePrefix(stems, coarse_poses)
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
