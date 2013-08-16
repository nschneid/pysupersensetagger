'''
Created on Jul 19, 2013

@author: Nathan Schneider (nschneid)
'''
from __future__ import print_function, division, absolute_import
import sys, os, re, fileinput, codecs, json
from collections import defaultdict, Counter
from heapq import heappush, heappop

import morph

_lexicons = {}
_lists = {}

def load_lexicons(lexfiles, is_list=False):
    for lexfile in lexfiles:
        name = os.path.split(lexfile.name.replace('.json',''))[-1]
        assert name not in _lexicons,(name,_lexicons.keys())
        print('loading', ('list' if is_list else 'lexicon'), name, end=' ', file=sys.stderr)
        resource = MultiwordLexicon(name, lexfile, is_list=is_list)
        (_lists if is_list else _lexicons)[name] = resource
        print(len(resource._entries), 'entries', file=sys.stderr)

def load_combined_lexicon(name, lexfiles, is_list=False):
    print('loading combined', ('list' if is_list else 'lexicon'), name, file=sys.stderr)
    combined = MultiwordLexicon(name, is_list=is_list)
    for lexfile in lexfiles:
        print('  loading file:', os.path.split(lexfile.name.replace('.json',''))[-1], file=sys.stderr)
        combined.loadJSON(lexfile)
    print(len(combined._entries), 'total entries', file=sys.stderr)
    (_lists if is_list else _lexicons)[name] = combined

def gappy_match(needle, haystack, start=0, max_gap_length=None):
    '''
    @param needle: Sequence of tokens to match, in order, but possibly with intervening gaps.
    @param haystack: Sequence of tokens to search for a match in.
    @param start: First token that is eligible for inclusion in the match.
    @param max_gap_length: Maximum number of words inside a gap. Unlimited by default. 
    
    If there are multiple matches (due to repeated words), smaller gaps are preferred.
    
    example result: [[2, 4, ['give', 'up']], [6, 7, ['on']]]
    '''
    h = ' '.join(haystack)
    if max_gap_length is not None:
        assert max_gap_length>=0
    gap_operator = '*' if max_gap_length is None else ('{,'+str(max_gap_length)+'}')
    pattern = r'(?:^|\s)(' + (r')(?:\s\S+)'+gap_operator+r'?\s(').join(re.escape(w) for w in needle) + ')$'
    #print(pattern, file=sys.stderr)
    m = re.search(pattern, h)
    if not m:
        return None
    result = []
    e = None
    for i,w in enumerate(m.groups()):
        itok = h[:m.start(i+1)].count(' ')
        if itok<start:
            return None
        if itok==e:
            result[-1][1] = itok+1
            result[-1][2].append(w)
        else:
            result.append([itok, itok+1, [w]])
        e = itok + 1
    return result

class MultiwordLexicon(object):
    '''
    The *signature* of an entry is the tuple of lemmas of 
    lexicalized parts of the expression, in order.
    '''

    POS_2_PENN = {'said': {',': ',', 
                       'A': 'JJ',    # not a perfect mapping due to VBD/VBG participles :P
                       'AP': 'JJ',
                       'AUX': 'VB',
                       'CONJ': 'CC', 
                       'COMP': {'that': 'WDT',
                                'which': 'WDT',
                                'whichever': 'WDT',
                                'what': 'WDT',
                                None: 'IN'},
                       'DEG': 'RB',
                       'DET': 'DT',
                       'MOD': 'MD',
                       'NEG': 'RB',
                       'N': 'NN',
                       'NP': 'NN',
                       'P': {'to': 'TO',
                             None: 'IN'},
                       'POSS': 'POS',
                       'PP': 'IN',
                       'PRON': 'PRP',
                       'Q': {'all': 'DT',
                             'some': 'DT',
                             'many': 'DT',
                             'few': 'DT',
                             'several': 'DT', 
                             'both': 'DT',
                             'no': 'DT',
                             None: 'CD'}, 
                       "S": 'VB',
                       "S'": 'VB',
                       'TO': 'TO',
                       'V': 'VB',
                       'VP': 'VB'
                       },
                'baldwin vpc': {'V': 'VB',
                                'P': {'to': 'TO',
                                      None: 'IN'}},
                'wikimwe': { # TreeTagger tagset
                            'NP': 'NNP', 'NPS': 'NNPS',
                            'VD': 'VB', 'VDD': 'VBD', 'VDG': 'VBG', 'VDN': 'VBN', 'VDP': 'VBP', 'VDZ': 'VBZ', # do
                            'VH': 'VB', 'VHD': 'VBD', 'VHG': 'VBG', 'VHN': 'VBN', 'VHP': 'VBP', 'VHZ': 'VBZ', # have
                            'VV': 'VB', 'VVD': 'VBD', 'VVG': 'VBG', 'VVN': 'VBN', 'VVP': 'VBP', 'VVZ': 'VBZ', # verb other than do, have, or be
                            'IN/that': 'IN'}}
    
    def __init__(self, name, jsonPath=None, is_list=False):
        self._name = name
        self._entries = {}
        self._bylast = defaultdict(set)
        self._is_list = is_list
        if jsonPath is not None:
            self.loadJSON(jsonPath)
    
    def _read_entry(self, entry):
        ds = entry["datasource"].lower()
        if ds=='wikimwe':
            words = entry["lemmas"] # not actually always lemmatized
            del entry["lemmas"]
            entry["words"] = words
        if ds in self.POS_2_PENN:   # map POSes to Penn Treebank tagset
            for i,p in enumerate(entry["poses"]):
                info = self.POS_2_PENN[ds].get(p,p)
                entry["poses"][i] = info if isinstance(info,basestring) else (info.get(entry["lemmas" if ds=='baldwin vpc' else "words"][i]) or info[None])
        
        
        if "lemmas" not in entry:
            
            if 'lvc' in ds:
                entry["lemmas"] = [entry["verblemma"], morph.stem(entry["noun"],'NN')]
            else:
                assert "words" in entry,entry
                words = entry["words"]
                poses = [None]*len(words)
                if "poses" in entry and entry["poses"]:
                    assert ds in {'said','semcor','wikimwe'},entry
                    poses = entry["poses"]
                elif ds in {'phrases.net', "oyz's idioms"}:
                    pass
                elif entry["label"].startswith('NNP') or entry["label"].startswith('NE:'):
                    poses = ['NNP']*len(words)
                entry["lemmas"] = [morph.stem(w,p) for w,p in zip(words,poses)]
        try:
            sig = tuple(l.lower() for l in entry["lemmas"] if not l[0]==l[-1]=='_')
            if not sig or sig[-1]=='the' or not any(l for l in sig if len(l)>2):
                return    # probably garbage entry
            if len(sig)>1:
                self._entries[sig] = entry
                self._bylast[sig[-1]].add(sig)
        except:
            print(entry, file=sys.stderr)
            raise
    
    def load(self, entries):
        iln = 1
        for entry in entries:
            self._read_entry(entry)
            iln += 1
        self._bylast = dict(self._bylast)   # convert from defaultdict
    
    def loadJSON(self, jsonF):
        iln = 1
        for ln in jsonF:
            entry = json.loads(ln[:-1].decode('utf-8'))
            if self._is_list:   # ranked list
                assert "rank" not in entry
                entry["rank"] = iln
            self._read_entry(entry)
            iln += 1
        self._bylast = dict(self._bylast)   # convert from defaultdict
    
    def __getitem__(self, signature):
        return self._entries[signature]
    
    def signatures_by_last_lemma(self, lemma):
        return self._bylast.get(lemma) or ()
    
    def shortest_path_decoding(self, sentence_lemmas, start=0, in_gap=False, max_gap_length=None):
        '''
        Use Dijkstra's algorithm to search a sentence from end to beginning 
        for a least-cost segmentation into lexical expressions according to 
        this lexicon. Each expression consists of a single word (token) 
        or a multiword unit from this lexicon. 
        
        Longer expressions are preferred over shorter ones. 
        Each top-level expression has a cost of 1; each expression within 
        a gap has a cost of 1.25. A recursive call with in_gap=True 
        will compute the least-cost contiguous segmentation nested within each gap.
        
        Example costs:
        
        a_b c (2) < a_ b _c (2.25) < a b c (3)
        
        a_b_c d (2) = a_b c_d (2) < a_ b _c_d (2.25) = a_ b_c _d (2.25) < a b_c d (3) 
          < a_ b _c d (3.25) < a_ b c _d (3.5) < a b c d (4)
        
        a_b_c d e (3) = a_b c_d e (3) < a_ b _c d_e (3.25) < a_ b _c_ d _e (3.5) 
          < a_b c d e (4) < a b c d e (5)
        '''
        # cost value is the number of edges in the path
        queue = []
        e = len(sentence_lemmas)
        assert 0<=start<e
        path = []
        tokinfo = []
        tags = ''
        while True:
            l = sentence_lemmas[e-1]
            
            # single-word option
            heappush(queue, (len(path)+1, e-1, e, [[l]]+path, ('o' if in_gap else 'O')+tags, [(e-1,('o' if in_gap else 'O'),(e-1,),False,None)]+tokinfo))
            #+e
            for cand in self.signatures_by_last_lemma(l):
                b = e-len(cand)
                if b<start: continue
                candL = list(cand)
                candinfo = self[cand]
                if sentence_lemmas[b:e]==candL:
                    newtags = 'B'+'I'*(len(cand)-1)
                    if in_gap:
                        newtags = newtags.lower()
                    myrange = tuple(range(b,e))
                    newtokinfo = [(b,('b' if in_gap else 'B'),myrange,False,candinfo)]+[(i,('i' if in_gap else 'I'),myrange,False,candinfo) for i in range(b+1,e)]
                    heappush(queue, (len(path)+1, b, e, [cand]+path, newtags+tags, newtokinfo+tokinfo))
                    #+b+1
                elif not in_gap and max_gap_length!=0 and set(sentence_lemmas[start:e])>=set(cand):
                    subspans = gappy_match(cand, sentence_lemmas[:e], start=start, max_gap_length=max_gap_length)
                    if subspans:
                        assert len(subspans)>1,subspans
                        myrange = tuple([i for a,b,c in subspans for i in range(a,b)])
                        subspans = subspans[::-1]
                        newtags = ''
                        newpath = []
                        newtokinfo = []
                        gapCost = 0
                        for before_gap,after_gap in zip(subspans[1:],subspans[:-1]):
                            gb = before_gap[1]
                            ge = after_gap[0]
                            assert ge>0, (cand, sentence_lemmas[:e], start, subspans)
                            gpath, gtags, gtokinfo = self.shortest_path_decoding(sentence_lemmas[:ge], start=gb, in_gap=True)
                            gapCost += 1.25*(len(gtags)-gtags.count('i'))  # 1.25 * count of in-gap lexical expressions
                            newpath = gpath + newpath
                            newtags = gtags + 'I'*(after_gap[1]-after_gap[0]) + newtags
                            newtokinfo = gtokinfo + [(i,'I',myrange,True,candinfo) for i in range(after_gap[0],after_gap[1])] + newtokinfo
                        newpath = [cand, newpath]
                        b = before_gap[0]
                        newtags = 'B' + 'I'*(before_gap[1]-b-1) + newtags
                        newtokinfo = [(b,'B',myrange,True,candinfo)] + [(i,'I',myrange,True,candinfo) for i in range(b+1,before_gap[1])] + newtokinfo
                        heappush(queue, (len(path)+1+gapCost, b, e, newpath+path, newtags+tags, newtokinfo+tokinfo))
                        #+b+1
            if not queue:
                raise Exception('Something went wrong: '+repr(sentence_lemmas))

            val, e, _, path, tags, tokinfo = heappop(queue)   # old beginning is the new end
            if e==start:
                # found a shortest path from the end to the specified start
                assert len(tags)==len(tokinfo)==len(sentence_lemmas[start:])
                return path, tags, tokinfo


def test():
    lex = MultiwordLexicon('Lex!')
    lex.load([{'lemmas': ['louis', 'xiv'], 'label': 'NE', 'datasource': '_'},
              {'lemmas': ['louis', 'armstrong'], 'label': 'NE', 'datasource': '_'},
              {'lemmas': ['neil', 'armstrong'], 'label': 'NE', 'datasource': '_'},
              {'lemmas': ['good', "ol'"], 'label': 'Idiom', 'datasource': '_'},
              {'lemmas': ['give', '_sb_', "_sb's_", 'due'], 'label': 'Idiom', 'datasource': '_'},
              {'lemmas': ['give', 'up', 'the', 'ghost'], 'label': 'Idiom', 'datasource': '_'},
              {'lemmas': ['give', 'up', 'the', 'ghost', 'on'], 'label': 'Idiom', 'datasource': '_'}, 
              {'lemmas': ['give', 'up', 'the', 'ghost', 'on', '_sb_'], 'label': 'Idiom', 'datasource': '_'}, 
              {'lemmas': ['give', 'up', 'on', '_sb_'], 'label': 'Idiom', 'datasource': '_'},
              {'lemmas': ['something', "'s", 'gotta', 'give'], 'label': 'Idiom', 'datasource': '_'}])
    print(lex._entries)
    print()
    print(lex._bylast)
    print()
    sentences = ["Something 's gotta give !".lower().split(),
                 "Do n't give up the ghost on Louis Armstrong .".lower().split(), 
                 "Do n't give up on Louis Armstrong .".lower().split(), 
                 "Do n't give Louis Armstrong up .".lower().split(), 
                 'You gotta give Louis Armstrong his due .'.lower().split(), 
                 "Louis Armstrong 's due must be give him .".lower().split(), 
                 'You gotta give old Louis Armstrong his due .'.lower().split(), 
                 "You gotta give good ol' Louis Armstrong his due .".lower().split(), 
                 "Louis Armstrong XIV".lower().split(), 
                 ]
    for sent in sentences:
        path, tags, tokinfo = lex.shortest_path_decoding(sent)
        assert tags==''.join(tag for toffset,tag,expr_tokens,is_gappy_expr,entry in tokinfo)
        print(path,tags,tokinfo)
        

if __name__=='__main__':
    test()
