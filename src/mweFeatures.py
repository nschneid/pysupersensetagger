'''
Ported from Michael Heilman's SuperSenseFeatureExtractor.java
(parts refactored into the 'morph' module).
@author: Nathan Schneider (nschneid)
@since: 2012-07-22
'''
from __future__ import print_function, division, absolute_import
import sys, os, re, gzip, codecs, json
from collections import Counter, defaultdict

from nltk.corpus import wordnet as wn

from pyutil.memoize import memoize
from pyutil.ds.features import IndexedFeatureMap
from pyutil.ds.trie import Trie
from pyutil.corpus import mwe_lexicons

SRCDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = SRCDIR+'/../data'

def hasFirstOrderFeatures():
    return True

def registerOpts(program_args):
    if program_args.lex is not None:
        mwe_lexicons.load_lexicons(program_args.lex)

"""
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
        mwe_lexicons.load_lexicons(program_args.lex)


def hasFirstOrderFeatures():
    return _options['usePrevLabel']

def wordClusterID(word):
    cid = clusterMap.get(word.lower(), 'UNK')
    if cid=='UNK': return cid
    return 'C'+str(cid)


clusterMap = None


startSymbol = None
endSymbol = None





        

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
"""

def extractLexiconCandidates(sent):
    '''
    For each lexicon, compute the shortest-path lexical segmentation 
    of the sentence under that lexicon. 
    Return a list of MWE membership information tuples for each token 
    according to that segmentation.
    '''
    sentence_lemmas = [t.stem for t in sent]
    return {lexiconname: lex.shortest_path_decoding(sentence_lemmas, max_gap_length=2)[2] 
            for lexiconname,lex in mwe_lexicons._lexicons.items()}


"""
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
"""



@memoize
def coarsen(pos):
    
    if pos=='TO': return 'I'
    elif pos.startswith('NNP'): return '^'
    elif pos=='CC': return '&'
    elif pos=='CD': return '#'
    elif pos=='RP': return 'T'
    else: return pos[0]

def isCompound(tok1, tok2):
    if tok1 is None or tok2 is None:
        return False
    l1 = tok1.stem
    l2 = tok2.stem
    ll = [l1,l2]
    return wn.lemmas(''.join(ll)) or wn.lemmas('_'.join(ll)) or wn.lemmas('-'.join(ll))

CPOS_PAIRS = [{'V','V'},{'V','N'},{'V','R'},{'V','T'},{'V','M'},{'V','P'},
              {'J','N'},{'N','N'},{'D','N'},{'D','^'},{'N','^'},{'^','^'},
              {'R','J'},{'N','&'},{'^','&'},{'V','I'},{'I','N'}]

DIGIT_RE = re.compile(r'\d')
SENSENUM = re.compile(r'\.(\d\d|XX)')

def extractFeatureValues(sent, j, usePredictedLabels=True, orders={0,1}, indexer=None,
                         lexiconCandidatesThisSent=None):
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
    
    
    ff = IndexedFeatureMap(indexer) if indexer is not None else {}
    
    
    # note: in the interest of efficiency, we use tuples rather than string concatenation for feature names
    
    # previous label feature (first-order Markov dependency)
    if 1 in orders and hasFirstOrderFeatures() and j>0:
            ff["prevLabel=",(sent[j-1].prediction if usePredictedLabels else sent[j-1].gold)] = 1
    
    if 0 in orders:
        # bias
        ff[()] = 1
        
        # first sense features
        # cluster features
         
        # original token, token position-in-sentence features
        if sent[j].token[0].isupper():
            ff['capitalized_BOS' if j==0 else 'capitalized_!BOS'] = 1
        ff['shape', sent[j].shape] = 1
        if j<2:
            ff['offset_in_sent=',str(j)] = 1
        if len(sent)-j<2:
            ff['offset_in_sent=',str(j-len(sent))] = 1
        
        # lowercased token features
        w = sent[j].token.lower()
        
        # - prefix (up to 4)
        # - suffix (up to 4)
        for k in range(4):
            ff['w[:{}]'.format(k+1), w[:k+1]] = 1
            ff['w[{}:]'.format(-k-1), w[-k-1:]] = 1
        
        # - special characters
        for c in w:
            if c.isdigit():
                ff['has-digit'] = 1
            elif not c.isalpha():
                ff['has-char', c] = 1
        
        # - context word up to 2 away
        # - context POS up to 2 words away
        # - context word bigram
        # - context POS bigram
        for k in range(j-2,j+3):
            if k<0: continue
            elif k>len(sent)-1: break
            ff['w_{:+}'.format(k-j), sent[k].token.lower()] = 1
            ff['pos_{:+}'.format(k-j), sent[k].pos] = 1
            if k<j+2 and k<len(sent)-1:
                ff['w_{:+},{:+}'.format(k-j,k-j+1), sent[k].token.lower(), sent[k+1].token.lower()] = 1
                ff['pos_{:+},{:+}'.format(k-j,k-j+1), sent[k].pos, sent[k+1].pos] = 1
        
        # - word + context POS
        # - POS + context word
        if j>0:
            ff['w_+0_pos_-1', sent[j].token.lower(), sent[j-1].pos] = 1
            ff['w_-1_pos_+0', sent[j-1].token.lower(), sent[j].pos] = 1
        if j<len(sent)-1:
            ff['w_+0_pos_+1', sent[j].token.lower(), sent[j+1].pos] = 1
            ff['w_+1_pos_+0', sent[j+1].token.lower(), sent[j].pos] = 1
        
        
        # lexicon features
        
        if not wn.lemmas(sent[j].stem):
            ff['OOV',sent[j].pos] = 1
            wn_pos_setS = '{}'
        else:
            wn_pos_set = frozenset({lem.synset.pos.replace('s','a') for lem in wn.lemmas(sent[j].stem)})
            wn_pos_setS = '{'+repr(tuple(wn_pos_set))[1:-1]+'}'
        
        
        # - compound
        if sent[j].pos.isalnum():
            prevtok = None
            for tok in sent[j-1::-1]:
                if tok.pos=='HYPH':
                    continue
                elif tok.pos.isalnum():
                    prevtok = tok
                break
            nexttok = None
            for tok in sent[j+1:]:
                if tok.pos=='HYPH':
                    continue
                elif tok.pos.isalnum():
                    nexttok = tok
                break
            
            if sent[j].pos=='HYPH':
                if isCompound(prevtok,nexttok):
                    ff['compound_left_right'] = 1
            else:
                if isCompound(prevtok,sent[j]):
                    ff['compound_left'] = 1
                if isCompound(sent[j],nexttok):
                    ff['compound_right'] = 1
        
        
        nMatches = Counter()
        for lexiconname,segmentation in lexiconCandidatesThisSent.items():
            toffset,tag,expr_tokens,is_gappy_expr,entry = segmentation[j]
            assert toffset==j
            if lexiconname=='wordnet_mwes':
                if entry:
                    try:
                        mw_pos_set = frozenset(wn.lemma(wnlemma).synset.pos.replace('s','a') for wnlemma in entry["wnlemmas"])
                    except:
                        print(entry, file=sys.stderr)
                        raise
                    mw_pos_setS = '{'+repr(tuple(mw_pos_set))[1:-1]+'}'
                    ff['wn',wn_pos_setS,tag,mw_pos_setS] = 1
                else:
                    ff['wn',wn_pos_setS,tag] = 1
            
            if tag.upper()!='O':
                lbl = entry["label"]
                if not lbl.startswith('NE:') and SENSENUM.search(lbl):
                    lbl = '<sense-tagged>'
                ff['lex',lexiconname,tag.upper(),str(is_gappy_expr),lbl] = 1
                if True or entry["datasource"].lower()!='wikimwe':
                    p1 = sent[expr_tokens[0]].pos
                    p2 = sent[expr_tokens[-1]].pos
                    ff['lex',lexiconname,tag.upper(),str(is_gappy_expr),lbl,p1,'...',p2] = 1
                    nMatches[p1,p2] += 1
                nMatches[None,None] += 1
            else:
                ff['lex',lexiconname,tag.upper()] = 1
            
        if nMatches[None,None]==0:
            ff['#lex-matches=','0'] = 1
        else:
            for n in range(1,nMatches[None,None]+1):
                ff['#lex-matches>=',str(n)] = 1
            for (p1,p2),N in nMatches.items():
                if (p1,p2)!=(None,None):
                    for n in range(1,N+1):
                        ff['#lex-matches',p1,'...',p2,'>=',str(n)] = 1
                
        # TODO: collocation
        
        sentpos = ''.join(coarsen(w.pos) for w in sent)
        cposj = coarsen(sent[j].pos)
        
        
    return ff
