#!/usr/bin/env python2.7
#coding=utf-8
'''
Measures MWE prediction performance.

Args: [--default-strength strong|weak] gold.mwe pred.mwe

@author: Nathan Schneider (nschneid@cs.cmu.edu)
@since: 2014-03-31
'''

from __future__ import print_function, division

import json, os, sys, fileinput, codecs, re
from collections import defaultdict, Counter, namedtuple

class Ratio(object):
    '''
    Fraction that prints both the ratio and the float value.
    fractions.Fraction reduces e.g. 378/399 to 18/19. We want to avoid this.
    '''
    def __init__(self, numerator, denominator):
        self._n = numerator
        self._d = denominator
    def __float__(self):
        return self._n / self._d if self._d!=0 else float('nan')
    def __str__(self):
        return '{}/{}={:.4f}'.format(self.numeratorS, self.denominatorS, float(self))
    __repr__ = __str__
    def __add__(self, v):
        if v==0:
            return self
        if isinstance(v,Ratio) and self._d==v._d:
            return Ratio(self._n + v._n, self._d)
        return float(self)+float(v)
    def __mul__(self, v):
        return Ratio(self._n * float(v), self._d)
    def __truediv__(self, v):
        return Ratio(self._n / float(v) if float(v)!=0 else float('nan'), self._d)
    __rmul__ = __mul__
    @property
    def numerator(self):
        return self._n
    @property
    def numeratorS(self):
        return ('{:.2f}' if isinstance(self._n, float) else '{}').format(self._n)
    @property
    def denominator(self):
        return self._d
    @property
    def denominatorS(self):
        return ('{:.2f}' if isinstance(self._d, float) else '{}').format(self._d)

def is_tag(t):
    return t in {'B','b','O','o','I','Ī','Ĩ','i','ī','ĩ'}

def f1(prec, rec):
    return 2*prec*rec/(prec+rec) if prec+rec>0 else float('nan')

def coarsen(ptbpos):
    p = ptbpos
    # similar but not quite identical to the Twitter POS tagset: adss $=possessive M=modal Y=symbol -=hyphen
    if p=='POS' or p.endswith('P$'): return '$'	# possessive
    elif p.startswith('PRP') or p.startswith('WP'): return 'O'
    elif p.startswith('NNP'): return '^'
    elif p.startswith('N'): return 'N'
    elif p.startswith('V'): return 'V'
    elif 'RB' in p: return 'R'
    elif p.startswith('JJ'): return 'A'
    elif p.endswith('DT'): return 'D'
    elif p in {'IN','TO'}: return 'P'
    else: return {'MD': 'M', 'RP': 'T', 'EX': 'X', 'CC': '&', 'CD': '#', '#': 'Y', '``': '"', "''": '"', 
	              '$': 'Y', 'FW': 'F', 'SYM': 'Y', 'LS': 'Y', 'AFX': 'G', 'GW': 'G', 'XX': 'G', 'UH': '!', 'HYPH': '-', 'ADD': '@'}.get(p,p)

RE_TAGGING = re.compile(r'^(O|B(o|b[iīĩ]+|[IĪĨ])*[IĪĨ]+)+$'.decode('utf-8'))

def require_valid_tagging(tagging, kind='tagging'):
    '''Verifies the chunking is valid.'''
    
    # method 1: check regex
    ex1 = ex2 = None
    
    try:
        assert RE_TAGGING.match(''.join(tagging).decode('utf-8')),kind+': '+''.join(tagging)
    except AssertionError as ex1:
        pass
    
    # method 2: tag-by-tag (should produce same result as regex)
    try:
        for t1,t2 in zip(tagging[:-1],tagging[1:]):
            if t1 in {'B'}:
                assert t2 in {'I','Ī','Ĩ','o','b','i','ī','ĩ'},kind+': '+''.join(tagging)
            if t1 in {'b'}:
                assert t2 in {'i','ī','ĩ'},kind+': '+''.join(tagging)
            if t2 in {'I','Ī','Ĩ','o','b','i','ī','ĩ'}:
                assert t1!='O',kind+': '+''.join(tagging)
            if t2 in {'i','ī','ĩ'}:
                assert t1 in {'b','i','ī','ĩ'},kind+': '+''.join(tagging)
            elif t2 in {'B','O'}:
                assert t1 in {'O','I','Ī','Ĩ'},kind+': '+''.join(tagging)
            elif t2=='b':
                assert t1 in {'B','I','Ī','Ĩ','o','i','ī','ĩ'},kind+': '+''.join(tagging)
            elif t2=='o':
                assert t1 in {'B','I','Ī','Ĩ','o','i','ī','ĩ'},kind+': '+''.join(tagging)
        assert tagging[0] in {'O','B'},kind+': '+''.join(tagging)
        assert tagging[-1] in {'O','I','Ī','Ĩ'},kind+': '+''.join(tagging)
    except AssertionError as ex2:
        pass
        
    assert (ex1 is None)==(ex2 is None),'Buggy checks! '+repr(ex1 is None)+' '+repr(ex2 is None)+' '+kind+': '+''.join(tagging)
    if ex1:
        raise ex1


def form_groups(links):
        """
        >>> form_groups([(1, 2), (3, 4), (2, 5), (6, 8), (4, 7)])==[{1,2,5},{3,4,7},{6,8}]
        True
        """
        groups = []
        groupMap = {} # offset -> group containing that offset
        for a,b in links:
            assert a is not None and b is not None,links
            assert b not in groups,'Links not sorted left-to-right: '+repr((a,b))
            if a not in groupMap: # start a new group
                groups.append({a})
                groupMap[a] = groups[-1]
            assert b not in groupMap[a],'Redunant link?: '+repr((a,b))
            groupMap[a].add(b)
            groupMap[b] = groupMap[a]
        return groups


goldmwetypes, predmwetypes = Counter(), Counter()

def eval_sent(sent, poses, genstats, sstats, wstats, bypos, indata=None, default_strength=None and '_' and '~'):
    '''default_strength: '_' or '~' (what to do in case of plain 'I' or 'i')'''
    def strength(tag):
        return {'I': None, 'Ī': '_', 'Ĩ': '~', 'i': None, 'ī': '_', 'ĩ': '~'}[tag]

    # verify the taggings are valid
    for k,kind in [(1,'gold'),(2,'pred')]:
        tags = zip(*sent)[k]
        require_valid_tagging(tags, kind=kind)
        
    if indata:
        gdata, pdata = indata
        genstats['Gold_#Groups_'] += len(gdata["_"])
        genstats['Gold_#Groups~'] += len(gdata["~"])
        genstats['Gold_#Groups'] += len(gdata["_"]) + len(gdata["~"])
        genstats['Gold_#GappyGroups'] += sum(1 for grp in gdata["_"]+gdata["~"] if max(grp)-min(grp)+1!=len(grp))
        if "lemmas" in gdata:
            for grp in gdata["_"]: goldmwetypes['_'.join(gdata["lemmas"][i-1] for i in grp)] += 1
            for grp in gdata["~"]: goldmwetypes['~'.join(gdata["lemmas"][i-1] for i in grp)] += 1
        genstats['Pred_#Groups_'] += len(pdata["_"])
        genstats['Pred_#Groups~'] += len(pdata["~"])
        genstats['Pred_#Groups'] += len(pdata["_"]) + len(pdata["~"])
        genstats['Pred_#GappyGroups'] += sum(1 for grp in pdata["_"]+pdata["~"] if max(grp)-min(grp)+1!=len(grp))
        for grp in pdata["_"]: predmwetypes['_'.join(pdata["lemmas"][i-1] for i in grp)] += 1
        for grp in pdata["~"]: predmwetypes['~'.join(pdata["lemmas"][i-1] for i in grp)] += 1

    glinks, plinks = [], []
    g_last_BI, p_last_BI = None, None
    g_last_bi, p_last_bi = None, None
    for i,(tkn,goldTag,predTag) in enumerate(sent):
        
        if goldTag!=predTag:
            genstats['incorrect'] += 1
        else:
            genstats['correct'] += 1
        
        if goldTag in {'I','Ī','Ĩ'}:
            glinks.append((g_last_BI, i, strength(goldTag) or default_strength))
            g_last_BI = i
        elif goldTag=='B':
            g_last_BI = i
        elif goldTag in {'i','ī','ĩ'}:
            glinks.append((g_last_bi, i, strength(goldTag) or default_strength))
            g_last_bi = i
        elif goldTag=='b':
            g_last_bi = i
        
        if goldTag in {'O','o'}:
            genstats['gold_Oo'] += 1
            if predTag in {'O', 'o'}:
                genstats['gold_pred_Oo'] += 1
        else:
            genstats['gold_non-Oo'] += 1
            if predTag not in {'O', 'o'}:
                genstats['gold_pred_non-Oo'] += 1
                if (goldTag in {'b','i','ī','ĩ'})==(predTag in {'b','i','ī','ĩ'}):
                    genstats['gold_pred_non-Oo_in-or-out-of-gap_match'] += 1
                if (goldTag in {'B','b'})==(predTag in {'B','b'}):
                    genstats['gold_pred_non-Oo_Bb-v-Ii_match'] += 1
                if goldTag in {'Ī','Ĩ','ī','ĩ'} and predTag in {'Ī','Ĩ','ī','ĩ'}:
                    genstats['gold_pred_Ii'] += 1
                    if (strength(goldTag) or default_strength)==strength(predTag) or default_strength:
                        genstats['gold_pred_Ii_strength_match'] += 1
        
        
        if predTag in {'I','Ī','Ĩ'}:
            plinks.append((p_last_BI, i, strength(predTag) or default_strength))
            p_last_BI = i
        elif predTag=='B':
            p_last_BI = i
        elif predTag in {'i','ī','ĩ'}:
            plinks.append((p_last_bi, i, strength(predTag) or default_strength))
            p_last_bi = i
        elif predTag=='b':
            p_last_bi = i
    
    assert all(s for a,b,s in glinks) and all(s for a,b,s in plinks),"Specify --default-strength if plain 'I' or 'i' tags occur in input"
    
    for d,stats in [('+',sstats), ('-',wstats)]:
        # for strengthened or weakened scores
        glinks1 = [(a,b) for a,b,s in glinks if d=='+' or s=='_']
        plinks1 = [(a,b) for a,b,s in plinks if d=='+' or s=='_']
        ggroups1 = form_groups(glinks1)
        pgroups1 = form_groups(plinks1)
        
        # soft matching (in terms of links)
        stats['PNumer'] += sum(1 for a,b in plinks1 if any(a in grp and b in grp for grp in ggroups1))
        stats['PDenom'] += len(plinks1)
        stats['CrossGapPNumer'] += sum((1 if b-a>1 else 0) for a,b in plinks1 if any(a in grp and b in grp for grp in ggroups1))
        stats['CrossGapPDenom'] += sum((1 if b-a>1 else 0) for a,b in plinks1)
        stats['RNumer'] += sum(1 for a,b in glinks1 if any(a in grp and b in grp for grp in pgroups1))
        stats['RDenom'] += len(glinks1)
        stats['CrossGapRNumer'] += sum((1 if b-a>1 else 0) for a,b in glinks1 if any(a in grp and b in grp for grp in pgroups1))
        stats['CrossGapRDenom'] += sum((1 if b-a>1 else 0) for a,b in glinks1)
        
        # exact matching (in terms of full groups)
        stats['ENumer'] += sum(1 for grp in pgroups1 if grp in ggroups1)
        stats['EPDenom'] += len(pgroups1)
        stats['ERDenom'] += len(ggroups1)
        
        for grp in ggroups1:
            posseq = tuple([coarsen(poses[i]) for i in sorted(grp)])
            gappiness = 'ng' if max(grp)-min(grp)+1==len(grp) else 'g'
            bypos[posseq][gappiness+d]['all'] += 1
            bypos[()][gappiness+d]['all'] += 1
            if grp in pgroups1:
                bypos[posseq][gappiness+d]['perfect'] += 1
                bypos[()][gappiness+d]['perfect'] += 1
            elif any(pgrp & grp for pgrp in pgroups1):
                bypos[posseq][gappiness+d]['partial'] += 1
                bypos[()][gappiness+d]['partial'] += 1
            else:
                bypos[posseq][gappiness+d]['miss'] += 1
                bypos[()][gappiness+d]['miss'] += 1
        for grp in pgroups1:
            gappiness = 'ng' if max(grp)-min(grp)+1==len(grp) else 'g'
            stats['Pred_'+gappiness] += 1
    
    
if __name__=='__main__':
    args = sys.argv[1:]
    default_strength = None
    if args[0]=='--default-strength':
        default_strength = {'strong': '_', 'weak': '~'}[args[1]]
        args = args[2:]

    genstats, sstats, wstats = Counter(), Counter(), Counter()
    bypos = defaultdict(lambda: {'ng-': Counter(), 'g-': Counter(), 'ng+': Counter(), 'g+': Counter()})
    
    sent = []
    goldFP, predFP = args
    predFP = fileinput.input(predFP)
    for gln in fileinput.input(goldFP):
        gparts = gln[:-1].split('\t')
        gdata = json.loads(gparts[2])
        gtags = [t[0].encode('utf-8') for t in gdata["tags"]]  # remove class labels, if any
        pln = next(predFP)
        pparts = pln[:-1].split('\t')
        pdata = json.loads(pparts[2])
        ptags = [t[0].encode('utf-8') for t in pdata["tags"]]  # remove class labels, if any
        words, poses = zip(*gdata["words"])
        eval_sent(zip(words,gtags,ptags), poses, 
                  genstats, sstats, wstats, bypos, indata=(gdata,pdata), default_strength=default_strength)
    
    nTags = genstats['correct']+genstats['incorrect']
    genstats['Acc'] = Ratio(genstats['correct'], nTags)
    genstats['Tag_R_Oo'] = Ratio(genstats['gold_pred_Oo'], genstats['gold_Oo'])
    genstats['Tag_R_non-Oo'] = Ratio(genstats['gold_pred_non-Oo'], genstats['gold_non-Oo'])
    genstats['Tag_Acc_non-Oo_in-gap'] = Ratio(genstats['gold_pred_non-Oo_in-or-out-of-gap_match'], genstats['gold_pred_non-Oo'])
    genstats['Tag_Acc_non-Oo_B-v-I'] = Ratio(genstats['gold_pred_non-Oo_Bb-v-Ii_match'], genstats['gold_pred_non-Oo'])
    genstats['Tag_Acc_I_strength'] = Ratio(genstats['gold_pred_Ii_strength_match'], genstats['gold_pred_Ii'])
    
    for stats in (sstats, wstats):
        stats['P'] = Ratio(stats['PNumer'], stats['PDenom'])
        stats['R'] = Ratio(stats['RNumer'], stats['RDenom'])
        stats['F'] = f1(stats['P'], stats['R'])
        stats['CrossGapP'] = stats['CrossGapPNumer']/stats['CrossGapPDenom'] if stats['CrossGapPDenom']>0 else float('nan')
        stats['CrossGapR'] = stats['CrossGapRNumer']/stats['CrossGapRDenom'] if stats['CrossGapRDenom']>0 else float('nan')
        stats['EP'] = Ratio(stats['ENumer'], stats['EPDenom'])
        stats['ER'] = Ratio(stats['ENumer'], stats['ERDenom'])
        stats['EF'] = f1(stats['EP'], stats['ER'])
    
    # (sstats+wstats)/2
    genstats += Counter({k: 0.5*v for k,v in (sstats + wstats).items()})
    
    genstats += Counter({k+'+': v for k,v in sstats.items()})
    genstats += Counter({k+'-': v for k,v in wstats.items()})
    
    if goldmwetypes:
        assert genstats['Gold_#Groups']==sum(goldmwetypes.values())
        genstats['Gold_#Types'] = len(goldmwetypes)
    assert genstats['Pred_#Groups']==sum(predmwetypes.values())
    genstats['Pred_#Types'] = len(predmwetypes)
    
    print(genstats)
    
    print(bypos)
    
    print(' '*12+'[- only]  missed | partial | perfect | gold count')
    print(' '*20+' all (gappy)')
    for posseq,counts in sorted(bypos.items(), key=lambda (k,v): v['ng-']['all']+v['g-']['all'], reverse=True):
        print('{:20}'.format(' '.join(posseq)) + '  '.join('{:>8}'.format('{} ({})'.format(counts['ng-'][criterion]+counts['g-'][criterion], counts['g-'][criterion])) \
            for criterion in ('miss', 'partial', 'perfect', 'all')))
    
    print('   P   |   R   |   F   |   EP  |   ER  |   EF  |  Acc  |   O   | non-O | ingap | B vs I | strength')
    parts = [(' {:.2%}'.format(float(genstats[x])), 
              '{:>7}'.format('' if isinstance(genstats[x],(float,int)) else genstats[x].numeratorS), 
              '{:>7}'.format('' if isinstance(genstats[x],(float,int)) else genstats[x].denominatorS)) for x in ('P', 'R', 'F', 'EP', 'ER', 'EF', 'Acc', 
              'Tag_R_Oo', 'Tag_R_non-Oo', 
              'Tag_Acc_non-Oo_in-gap', 'Tag_Acc_non-Oo_B-v-I', 'Tag_Acc_I_strength')]
    for pp in zip(*parts):
        print(' '.join(pp))
        
    #print(predmwetypes)