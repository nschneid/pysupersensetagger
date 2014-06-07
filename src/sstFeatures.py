'''
Feature extraction routines for joint supersense tagging + MWE identification.
Adds a few new features to the TACL 2014 MWE model (mweFeatures.py), 
most notably consulting a supersense lexicon derived from WordNet.
For an old feature extractor that reimplemented those of Ciaramita & Altun (2006)'s 
supersense tagger, see supersenseFeatureExtractor.py.

@author: Nathan Schneider (nschneid)
@since: 2014-06-06
'''
from __future__ import print_function, division, absolute_import
import sys, os, re, gzip, codecs, json
from collections import Counter, defaultdict

from nltk.corpus import wordnet as wn

from pyutil.memoize import memoize
from pyutil.ds.beam import Beam
from pyutil.ds.features import IndexedFeatureMap
from pyutil.ds.trie import Trie
from pyutil.corpus import mwe_lexicons

SRCDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = SRCDIR+'/../data'
LEXDIR = SRCDIR+'/../lex'

def hasFirstOrderFeatures():
    return True

clusterMap = None
topClusterMembers = defaultdict(lambda: Beam(3))

def registerOpts(program_args):
    if program_args.lex is not None:
        mwe_lexicons.load_lexicons(program_args.lex)
    if program_args.clist is not None:
        #mwe_lexicons.load_combined_lexicon('mwetoolkit YelpAcademic', [f for f in program_args.clist if f.name.startswith('mwetk.yelpAcademic')], is_list=True)
        #mwe_lexicons.load_lexicons([f for f in program_args.clist if not f.name.startswith('mwetk.yelpAcademic')], is_list=True)
        mwe_lexicons.load_lexicons(program_args.clist, is_list=True)
    if program_args.clusters:
        loadClusters(program_args.cluster_file)
    global useTokenBigrams, useWNOOV, useWNCompound
    useTokenBigrams = not program_args.no_bigrams
    useWNOOV = not program_args.no_oov
    useWNCompound = not program_args.no_compound

def loadClusters(clusterFile, oldClusterFormat=False):
    global clusterMap
    clusterMap = {}
    
    print("loading word clusters...", file=sys.stderr);
    with gzip.open(clusterFile) as clusterF:
        if oldClusterFormat:    # each line is a cluster, with space-separated words
            clusterID = 0
            for ln in clusterF:
                ww = re.split(r'\s', ln)
                for w in ww:
                    clusterMap[w] = clusterID
                    topClusterMembers[clusterID][w] = 1
                clusterID += 1
        else:   # each line contains a cluster ID (bitstring for Brown clusters), a word type, and a count
            for ln in clusterF:
                clusterID, w, n = ln[:-1].split('\t')
                w = w.decode('utf-8')
                clusterMap[w] = clusterID
                topClusterMembers[clusterID][w] = int(n)
    
    for k in topClusterMembers.keys():
        topClusterMembers[k] = '_'.join(topClusterMembers[k].keys())
        
    print("done.", file=sys.stderr);

@memoize
def wordClusterID(word):
    cid = clusterMap.get(word.lower(), 'UNK')
    if cid=='UNK': return cid, None
    return 'C'+str(cid), topClusterMembers[cid]


SUPERSENSES = list(map(str.strip, '''
noun.person         
noun.artifact         
noun.act
noun.cognition
noun.communication
noun.group         
noun.attribute     
noun.location
noun.time    
noun.state
noun.body    
noun.substance     
noun.quantity         
noun.event         
noun.object
noun.possession
noun.phenomenon
noun.animal
noun.relation
noun.feeling
noun.food
noun.process
noun.plant
noun.shape
noun.motive
noun.other
verb.stative         
verb.communication 
verb.change        
verb.cognition     
verb.social         
verb.motion         
verb.possession     
verb.contact         
verb.perception     
verb.creation      
verb.emotion         
verb.consumption     
verb.body
verb.competition
verb.weather
'''.strip().split()))
assert len(SUPERSENSES)==41,len(SUPERSENSES)
for i in range(len(SUPERSENSES)):
    pos, name = SUPERSENSES[i].split('.')
    if pos=='noun':
        name = name.upper()
    SUPERSENSES[i] = name.replace('OBJECT','NATURAL OBJECT')

NOUNTOPS = {'entity.n.01': 'OTHER',
'physical_entity.n.01': 'OTHER',
'abstraction.n.06': 'OTHER',
'thing.n.12': 'OTHER',
'object.n.01': 'OTHER',
'whole.n.02': 'OTHER',
'congener.n.03': 'OTHER',
'living_thing.n.01': 'OTHER',
'organism.n.01': 'OTHER',
'benthos.n.02': 'OTHER',
'dwarf.n.03': 'OTHER',
'heterotroph.n.01': 'OTHER',
'parent.n.02': 'OTHER',
'life.n.10': 'OTHER',
'biont.n.01': 'OTHER',
'cell.n.02': 'OTHER',
'causal_agent.n.01': 'OTHER',
'person.n.01': 'PERSON',
'animal.n.01': 'ANIMAL',
'plant.n.02': 'PLANT',
'native.n.03': 'OTHER',
'natural_object.n.01': 'NATURAL OBJECT',
'substance.n.01': 'SUBSTANCE',
'substance.n.07': 'SUBSTANCE',
'matter.n.03': 'OTHER',
'food.n.01': 'FOOD',
'nutrient.n.02': 'SUBSTANCE',
'artifact.n.01': 'ARTIFACT',
'article.n.02': 'ARTIFACT',
'psychological_feature.n.01': 'OTHER',
'cognition.n.01': 'COGNITION',
'motivation.n.01': 'MOTIVE',
'attribute.n.02': 'ATTRIBUTE',
'state.n.02': 'STATE',
'feeling.n.01': 'FEELING',
'location.n.01': 'LOCATION',
'shape.n.02': 'SHAPE',
'time.n.05': 'TIME',
'space.n.01': 'ATTRIBUTE',
'absolute_space.n.01': 'ATTRIBUTE',
'phase_space.n.01': 'ATTRIBUTE',
'event.n.01': 'EVENT',
'process.n.06': 'PROCESS',
'act.n.02': 'ACT',
'group.n.01': 'GROUP',
'relation.n.01': 'RELATION',
'possession.n.02': 'POSSESSION',
'social_relation.n.01': 'RELATION',
'communication.n.02': 'COMMUNICATION',
'measure.n.02': 'QUANTITY',
'phenomenon.n.01': 'PHENOMENON'}
assert not set(NOUNTOPS.values())-set(SUPERSENSES),set(NOUNTOPS.values())-set(SUPERSENSES)

@memoize
def supersense(synset):
    if synset.lexname=='noun.Tops':
        return NOUNTOPS[synset.name]
    pos, name = synset.lexname.split('.')
    if pos=='noun':
        return name.upper()
    elif pos=='verb':
        return name
    return synset.lexname   # adj.all, adv.all


senseTrie = Trie()  # lemma sequence -> ordered list of supersenses
senseCountMap = {} # pos -> {word -> number of senses}
possibleSensesMap = {} # stem -> set of possible supersenses

nSupersenseEntries = 0
if __name__=='__main__' and not os.path.exists(LEXDIR+'/wordnet_supersenses.json'):
    print('building WordNet supersense lexicon...', file=sys.stderr, end=' ')
    with open(LEXDIR+'/wordnet_supersenses.json','w') as outF:
        
        for lemma_name in wn.all_lemma_names():
            entry = {"lemma_name": lemma_name, "lemmas": lemma_name.split('_'), 
                     "supersenses": [supersense(syn) for syn in wn.synsets(lemma_name)]}
            nSupersenseEntries += 1
            outF.write(json.dumps(entry)+'\n')
    print('done:',nSupersenseEntries,'entries', file=sys.stderr)
if __name__!='__main__':
    print('loading WordNet supersense lexicon...', file=sys.stderr, end=' ')
    with open(LEXDIR+'/wordnet_supersenses.json') as inF:
        for ln in inF:
            entry = json.loads(ln.strip())
            allsupersenses = entry["supersenses"]
            supersenses = {"v": [sst for sst in allsupersenses if sst.islower() and '.' not in sst],
             "n": [sst for sst in allsupersenses if sst.isupper() and '.' not in sst]}
            senseTrie[entry["lemmas"]] = supersenses
            nSupersenseEntries += 1
        print('done:',nSupersenseEntries,'entries', file=sys.stderr)

@memoize
def _isO(label):
    return int(label[0].upper()=='O')

@memoize
def _isBI(label):
    return int(label[0].upper()!='O')

def extractWNSupersenseCandidates(sent):
    sw_supersenses = [()]*len(sent)   # WordNet supersenses associated with single word
    mw_supersenses = [()]*len(sent)   # WordNet supersenses associated with longest sequence beginning at the given word
    nextN_supersenses = [(None,())]*len(sent)    # WordNet supersenses associated with subsequent noun
    cposes = [coarsen(tok.pos) for tok in sent]
    stems = [tok.stem for tok in sent]
    for j,tok in enumerate(sent):
        # single-word supersenses
        supersenses = senseTrie.get([tok.stem])
        if supersenses:
            cpos = cposes[j].replace('^','N')
            if cpos in ('N','V'):
                sw_supersenses[j] = supersenses[cpos.lower()]
        
        stemMatch = None
        if cposes[j]=='V' and 'T' in cposes[j+1:]:  # verb followed by particle with no intervening verb
            k = cposes.index('T',j+1)
            if 'V' not in cposes[j+1:k]:
                stemMatch = senseTrie.longest([stems[j],stems[k]])
        if not stemMatch:
            stemMatch = senseTrie.longest(stems[j:])    # longest contiguous match
        if stemMatch:
            match, supersenses = stemMatch
            lastCPOS = cposes[j+len(match)-1]
            if lastCPOS in ('V','T') and supersenses["v"]:
                mw_supersenses[j] = supersenses["v"]
            else:
                mw_supersenses[j] = supersenses["n"]+supersenses["v"]
    
    for j,tok in enumerate(sent):
        cpos = cposes[j]
        if cpos in ('N','V','J') and 'N' in cposes[j+1:]:   # common noun, verb, adj
            # find next common noun with no intervening verb
            k = cposes.index('N',j+1)
            if 'V' not in cposes[j+1:k]:
                nextN_supersenses[j] = cpos,sw_supersenses[k]
                
    return sw_supersenses, mw_supersenses, nextN_supersenses

def extractWNSupersenseFeat(ff, j, supersenseCandidatesThisSent):
    
    sw_supersenses, mw_supersenses, (cpos,nextN_supersenses) = zip(*supersenseCandidatesThisSent)[j]
    
    #@memoize
    def _hasWNSupersense(label):
        parts = label.split('-')
        if len(parts)==1: return 0
        a, b = parts
        return int(b in (sw_supersenses if a.upper()=='O' else mw_supersenses))
    
    if not sw_supersenses and not mw_supersenses:
        ff['WN_supersense_unavailable'] = 1
        return
    
    # for [Oo]-* labels (single-word expression)
    firstO = sw_supersenses[0] if sw_supersenses else None
    # for [^Oo]-* labels (possibly multiword expression)
    firstBI = mw_supersenses[0] if mw_supersenses else None
    if firstO==firstBI:
        if firstO: ff['WN_1st_supersense',firstO] = 1
    else:
        if firstO: ff['WN_1st_supersense',firstO] = _isO
        if firstBI: ff['WN_1st_supersense',firstBI] = _isBI
    ff['WN_has_supersense'] = _hasWNSupersense
    if nextN_supersenses:
        ff['cpos',cpos,'WN_nextN_1st_supersense',nextN_supersenses[0]] = 1

def extractLexiconCandidates(sent):
    '''
    For each lexicon and collocation list, compute the shortest-path lexical segmentation 
    of the sentence under that lexicon. 
    Return a list of MWE membership information tuples for each token 
    according to that segmentation.
    '''
    #assert mwe_lexicons._lexicons   # actually, depends on whether any --lex args are present...
    sentence_lemmas = [t.stem for t in sent]
    return ({lexiconname: lex.shortest_path_decoding(sentence_lemmas, max_gap_length=2)[2] 
            for lexiconname,lex in mwe_lexicons._lexicons.items()}, 
            {listname: lex.shortest_path_decoding(sentence_lemmas, max_gap_length=2)[2] 
            for listname,lex in mwe_lexicons._lists.items()})




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

THRESHOLDS = [25,50,75,100,150]+range(200,1000,100)+range(10**3,10**4,10**3)+range(10**4,10**5,10**4)+range(10**5,10**6,10**5)+range(10**6,10**7,10**6)



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
    
    (lexiconCandidates, listCandidates), supersenseCandidates = candidatesThisSentence or (({}, {}), [])
    
    ff = IndexedFeatureMap(indexer) if indexer is not None else {}
    
    # note: in the interest of efficiency, we use tuples rather than string concatenation for feature names
    
    # previous label feature (first-order Markov dependency)
    if 1 in orders and hasFirstOrderFeatures() and j>0:
            ff["prevLabel=",(sent[j-1].prediction if usePredictedLabels else sent[j-1].gold)] = 1
    
    if 0 in orders:
        # bias
        ff[()] = 1
        
        
         
        # original token, token position-in-sentence features
        if sent[j].token[0].isupper():
            #ff['capitalized_BOS' if j==0 else 'capitalized_!BOS'] = 1 # old version of feature (in mweFeatures)
            nCap = sum(1 for tkn in sent if tkn.token[0].isupper())
            if j==0:
                ff['capitalized_BOS'] = 1
                if nCap>=(len(sent)-nCap):
                    ff['capitalized_BOS_majcap'] = 1
            else:
                ff['capitalized_!BOS'] = 1
                if nCap>=(len(sent)-nCap):
                    ff['capitalized_!BOS_majcap'] = 1
                if sent[0].token[0].islower():
                    ff['capitalized_!BOS_BOSlower'] = 1
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
        # - current lemma and context lemma up to 2 words away, if one of them is a verb 
        #   and the other is a noun, verb, adjective, adverb, preposition, or particle
        for k in range(j-2,j+3):
            if k<0: continue
            elif k>len(sent)-1: break
            ff['w_{:+}'.format(k-j), sent[k].token.lower()] = 1
            ff['pos_{:+}'.format(k-j), sent[k].pos] = 1
            if k!=j and ( \
                    (sent[k].pos[0]=='V' and sent[j].pos[0] in {'V','N','J','I','R','T'}) \
                 or (sent[j].pos[0]=='V' and sent[k].pos[0] in {'V','N','J','I','R','T'})):
                    ff['lemma_+0,{:+}'.format(k-j), sent[j].stem, sent[k].stem] = 1
            if k<j+2 and k<len(sent)-1:
                if useTokenBigrams: ff['w_{:+},{:+}'.format(k-j,k-j+1), sent[k].token.lower(), sent[k+1].token.lower()] = 1
                ff['pos_{:+},{:+}'.format(k-j,k-j+1), sent[k].pos, sent[k+1].pos] = 1
            if clusterMap and (k==j or abs(k-j)==1): # current and neighbor clusters
                clustid, keywords = wordClusterID(sent[k].token.lower())
                ff['c_{:+1}'.format(k-j), clustid, keywords or ''] = 1
                if k!=j:
                    ff['lemma_+0,c_{:+}'.format(k-j), sent[j].stem, clustid, keywords or ''] = 1
        
        # - word + context POS
        # - POS + context word
        if j>0:
            ff['w_+0_pos_-1', sent[j].token.lower(), sent[j-1].pos] = 1
            ff['w_-1_pos_+0', sent[j-1].token.lower(), sent[j].pos] = 1
        if j<len(sent)-1:
            ff['w_+0_pos_+1', sent[j].token.lower(), sent[j+1].pos] = 1
            ff['w_+1_pos_+0', sent[j+1].token.lower(), sent[j].pos] = 1
        
        
        # - auxiliary verb/main verb (new relative to mweFeatures)
        if coarsen(sent[j].pos)=='V':
            cposes = [coarsen(tok.pos) for tok in sent[j:]]
            if len(cposes)>1 and cposes[1]=='V':
                # followed by another verb: probably an aux (though there are exceptions: 
                # "try giving", "all people want is", etc.)
                ff['auxverb'] = 1
            elif len(cposes)>2 and cposes[1]=='R' and cposes[2]=='V':
                # followed by an adverb followed by a verb: probably an aux
                ff['auxverb'] = 1
            else:
                ff['mainverb'] = 1
        
        
        # lexicon features
        
        if not wn.lemmas(sent[j].stem):
            if useWNOOV: ff['OOV',sent[j].pos] = 1
            wn_pos_setS = '{}'
        else:
            wn_pos_set = frozenset({lem.synset.pos.replace('s','a') for lem in wn.lemmas(sent[j].stem)})
            wn_pos_setS = '{'+repr(tuple(wn_pos_set))[1:-1]+'}'
        
        # - WordNet supersense (new relative to mweFeatures)
        extractWNSupersenseFeat(ff, j, supersenseCandidates)
        
        if useWNCompound:
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
        for lexiconname,segmentation in lexiconCandidates.items():
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
                if True or entry["datasource"].lower()!='wikimwe':   # TODO: OK to remove constraint for wikimwe?
                    p1 = sent[expr_tokens[0]].pos
                    p2 = sent[expr_tokens[-1]].pos
                    ff['lex',lexiconname,tag.upper(),str(is_gappy_expr),lbl,p1,'...',p2] = 1
                    nMatches[p1,p2] += 1
                nMatches[None,None] += 1
            else:
                ff['lex',lexiconname,'O'] = 1
            
        if nMatches[None,None]==0:
            ff['#lex-matches=','0'] = 1
        else:
            for n in range(1,nMatches[None,None]+1):
                ff['#lex-matches>=',str(n)] = 1
            for (p1,p2),N in nMatches.items():
                if (p1,p2)!=(None,None):
                    for n in range(1,N+1):
                        ff['#lex-matches',p1,'...',p2,'>=',str(n)] = 1
        
        #sentpos = ''.join(coarsen(w.pos) for w in sent)
        #cposj = coarsen(sent[j].pos)
        
        
        # - collocation extraction lists
        # lists for 6 collocation classes: adj-noun noun-noun preposition-noun verb-noun verb-preposition verb-particle 
        # each list ranks lemma pairs using the t-test.
        # considering each list separately, we segment the sentence preferring higher-ranked items 
        # (requiring lemmas and coarse POSes to match). 
        # fire features indicating (a) B vs. I match, and (b) whether the rank in the top 
        # {25,50,75,100,150,200,300,...,900,1000,2000,...,9000,10k,20k,...90k,100k,200k,...}, 
        # (c) gappiness?
        
        
        for listname,segmentation in listCandidates.items():
            toffset,tag,expr_tokens,is_gappy_expr,entry = segmentation[j]
            assert toffset==j
            
            if tag.upper()!='O':
                lbl = entry["label"]
                is_phrasinator = (entry["datasource"].lower().startswith('phrasinator'))
                ff['list',listname,tag.upper(),str(is_gappy_expr),lbl] = 1
                
                
                p1 = sent[expr_tokens[0]].pos
                p2 = sent[expr_tokens[-1]].pos
                if is_phrasinator:
                    ff['list',listname,tag.upper(),str(is_gappy_expr),lbl,p1,'...',p2] = 1
                r = entry["rank"]
                for t in THRESHOLDS:
                    if r>t: break
                    ff['list',listname,'rank<={}'.format(t), tag.upper(),str(is_gappy_expr),lbl] = 1
                    if is_phrasinator:
                        ff['list',listname,'rank<={}'.format(t), tag.upper(),str(is_gappy_expr),lbl,p1,'...',p2] = 1
                
            else:
                ff['list',listname,'O'] = 1
                
    return ff
