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

def load_lexicons(lexfiles):
    for lexfile in lexfiles:
        name = os.path.split(lexfile.name.replace('.json',''))[-1]
        assert lexfile not in _lexicons
        _lexicons[name] = MultiwordLexicon(name, lexfile)
        

def gappy_match(needle, haystack, start=0):
    '''
    @param needle: Sequence of tokens to match, in order, but possibly with intervening gaps.
    @param haystack: Sequence of tokens to search for a match in.
    @param start: First token that is eligible for inclusion in the match.
    
    If there are multiple matches (due to repeated words), smaller gaps are preferred.
    
    example result: [[2, 4, ['give', 'up']], [6, 7, ['on']]]
    '''
    h = ' '.join(haystack)
    pattern = r'\b(' + r')(?:\s\S+)*?\s('.join(re.escape(w) for w in needle) + ')$'
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
    def __init__(self, name, jsonPath=None):
        self._name = name
        self._entries = {}
        self._bylast = defaultdict(set)
        if jsonPath is not None:
            self.loadJSON(jsonPath)
    
    def _read_entry(self, entry):
        if "lemmas" not in entry:
            assert "words" in entry,entry
            words = entry["words"]
            poses = [None]*len(words)
            if "poses" in entry and entry["poses"]:
                assert entry["datasource"].lower()=='semcor',entry
                poses = entry["poses"]
            elif entry["label"].startswith('NNP') or entry["label"].startswith('NE:'):
                poses = ['NNP']*len(words)
            entry["lemmas"] = [morph.stem(w,p) for w,p in zip(words,poses)]
        try:
            sig = tuple(l.lower() for l in entry["lemmas"] if not l[0]==l[-1]=='_')
        except:
            print(entry)
            raise
        if sig[-1]=='the' or not any(l for l in sig if len(l)>2):
            return    # probably garbage entry
        if len(sig)>1:
            self._entries[sig] = entry
            self._bylast[sig[-1]].add(sig)
    
    def load(self, entries):
        for entry in entries:
            self._read_entry(entry)
    
    def loadJSON(self, jsonF):
        for ln in jsonF:
            entry = json.loads(ln[:-1].decode('utf-8'))
            self._read_entry(entry)
    
    def __getitem__(self, signature):
        return self._entries[signature]
    
    def signatures_by_last_lemma(self, lemma):
        return self._bylast[lemma]
    
    def shortest_path_decoding(self, sentence_lemmas, start=0, in_gap=False):
        '''
        Use Dijkstra's algorithm to search a sentence from end to beginning 
        for a least-cost segmentation into lexical expressions according to 
        this lexicon. Each expression consists of a single word (token) 
        or a multiword unit from this lexicon. 
        
        Longer expressions are preferred over shorter ones. 
        Each expression has a cost of 1; additionally, gappy expressions 
        incur a cost of 1 for each gap. A recursive call with in_gap=True 
        will compute the least-cost contiguous segmentation nested within each gap 
        (this will not contribute to the cost of the "outer" segmentation).
        '''
        # cost value is the number of edges in the path
        queue = []
        e = len(sentence_lemmas)
        assert 0<=start<e
        est_val = float('inf')  # estimated cost (A* heuristic value)
        path = []
        tokinfo = []
        tags = ''
        while True:
            l = sentence_lemmas[e-1]
            
            # single-word option
            heappush(queue, (len(path)+e, e-1, e, [[l]]+path, ('o' if in_gap else 'O')+tags, [(e-1,('o' if in_gap else 'O'),False,None)]+tokinfo))
            
            for cand in self.signatures_by_last_lemma(l):
                b = e-len(cand)
                if b<start: continue
                candL = list(cand)
                candinfo = self[cand]
                if sentence_lemmas[b:e]==candL:
                    newtags = 'B'+'I'*(len(cand)-1)
                    if in_gap:
                        newtags = newtags.lower()
                    newtokinfo = [(b,('b' if in_gap else 'B'),False,candinfo)]+[(i,('i' if in_gap else 'I'),False,candinfo) for i in range(b+1,e)]
                    heappush(queue, (len(path)+b+1, b, e, [cand]+path, newtags+tags, newtokinfo+tokinfo))
                elif not in_gap:
                    subspans = gappy_match(cand, sentence_lemmas[:e], start=start)
                    if subspans:
                        assert len(subspans)>1,subspans
                        subspans = subspans[::-1]
                        newtags = ''
                        newpath = []
                        newtokinfo = []
                        for before_gap,after_gap in zip(subspans[1:],subspans[:-1]):
                            gb = before_gap[1]
                            ge = after_gap[0]
                            assert ge>0
                            gpath, gtags, gtokinfo = self.shortest_path_decoding(sentence_lemmas[:ge], start=gb, in_gap=True)
                            newpath = gpath + newpath
                            newtags = gtags + 'I'*(after_gap[1]-after_gap[0]) + newtags
                            newtokinfo = gtokinfo + [(i,'I',True,candinfo) for i in range(after_gap[0],after_gap[1])] + newtokinfo
                        newpath = [cand, newpath]
                        b = before_gap[0]
                        newtags = 'B' + 'I'*(before_gap[1]-b-1) + newtags
                        newtokinfo = [(b,'B',True,candinfo)] + [(i,'I',True,candinfo) for i in range(b+1,before_gap[1])] + newtokinfo
                        # the cost of a gappy expression is 1 + the number of gaps
                        heappush(queue, (len(path)+b+1, b, e, newpath+path, newtags+tags, newtokinfo+tokinfo))
            
            if not queue:
                raise Exception('Something went wrong: '+repr(sentence_lemmas))

            est_val, e, _, path, tags, tokinfo = heappop(queue)   # old beginning is the new end
            if e==start:
                # found a shortest path from the end to the specified start
                assert len(tags)==len(tokinfo)==len(sentence_lemmas[start:])
                return path, tags, tokinfo


def test():
    lex = MultiwordLexicon('Lex!')
    lex.load([{'lemmas': ['louis', 'xiv'], 'label': 'NE'},
              {'lemmas': ['louis', 'armstrong'], 'label': 'NE'},
              {'lemmas': ['neil', 'armstrong'], 'label': 'NE'},
              {'lemmas': ['good', "ol'"], 'label': 'Idiom'},
              {'lemmas': ['give', '_sb_', "_sb's_", 'due'], 'label': 'Idiom'},
              {'lemmas': ['give', 'up', 'the', 'ghost'], 'label': 'Idiom'},
              {'lemmas': ['give', 'up', 'the', 'ghost', 'on'], 'label': 'Idiom'}, 
              {'lemmas': ['give', 'up', 'the', 'ghost', 'on', '_sb_'], 'label': 'Idiom'}, 
              {'lemmas': ['give', 'up', 'on', '_sb_'], 'label': 'Idiom'},
              {'lemmas': ['something', "'s", 'gotta', 'give'], 'label': 'Idiom'}])
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
                 ]
    for sent in sentences:
        path, tags, tokinfo = lex.shortest_path_decoding(sent)
        assert tags==''.join(tag for toffset,tag,gappy_expr,entry in tokinfo)
        print(path,tags,tokinfo)
        

if __name__=='__main__':
    test()
