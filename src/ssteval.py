#!/usr/bin/env python2.7
#coding=utf-8
'''
Measures SST prediction performance, ignoring chunking (except for exact tag accuracy).
See mweval.py for measuring chunking performance.

Args: gold.sst pred.sst

@author: Nathan Schneider (nschneid@cs.cmu.edu)
@since: 2014-06-08
'''

from __future__ import print_function, division

import json, os, sys, fileinput, codecs, re
from collections import defaultdict, Counter, namedtuple

from mweval import Ratio, f1

def sstpos(sst):
    if '`' in sst: return '`'
    elif sst.isupper(): return 'N'
    elif sst[0].isupper(): return 'P'
    elif sst.islower(): return 'V'
    assert False,sst

if __name__=='__main__':
    args = sys.argv[1:]
    
    stats = defaultdict(Counter)
    conf = Counter()    # confusion matrix
    
    sent = []
    goldFP, predFP = args
    predFP = fileinput.input(predFP)
    for gln in fileinput.input(goldFP):
        gparts = gln[:-1].split('\t')
        gdata = json.loads(gparts[2])
        gtags = gdata["tags"]
        pln = next(predFP)
        pparts = pln[:-1].split('\t')
        pdata = json.loads(pparts[2])
        ptags = pdata["tags"]
        assert len(gtags)==len(ptags)
        words, poses = zip(*gdata["words"])
        
        stats['Exact Tag']['nGold'] += len(gtags)
        stats['Exact Tag']['tp'] += sum(1 for g,p in zip(gtags,ptags) if g==p)
        
        # now remove the positional indicator (B, I, O, etc.), leaving only the class
        gtags = [(g+'-').split('-')[1] for g in gtags]
        ptags = [(p+'-').split('-')[1] for p in ptags]
        
        # includes *-`a, excludes tags with no class (plain B, I, O, etc.)
        stats['All Classes']['nGold'] += sum(1 for g in gtags if g)
        stats['All Classes']['nPred'] += sum(1 for p in ptags if p)
        stats['All Classes']['tp'] += sum(1 for g,p in zip(gtags,ptags) if g and g==p)
        
        # collapse all class labels together
        stats['Any Class']['nGold'] += sum(1 for g in gtags if g)
        stats['Any Class']['nPred'] += sum(1 for p in ptags if p)
        stats['Any Class']['tp'] += sum(1 for g,p in zip(gtags,ptags) if g and p)
        
        # only supersense labels (not `a)
        stats['All SSTs']['nGold'] += sum(1 for g in gtags if g and '`' not in g)
        stats['All SSTs']['nPred'] += sum(1 for p in ptags if p and '`' not in p)
        stats['All SSTs']['tp'] += sum(1 for g,p in zip(gtags,ptags) if p and '`' not in p and g==p)

        # noun supersense labels
        stats['Noun SSTs']['nGold'] += sum(1 for g in gtags if g and g.isupper())
        stats['Noun SSTs']['nPred'] += sum(1 for p in ptags if p and p.isupper())
        stats['Noun SSTs']['tp'] += sum(1 for g,p in zip(gtags,ptags) if p and p.isupper() and g==p)
        
        # verb supersense labels (not `a)
        stats['Verb SSTs']['nGold'] += sum(1 for g in gtags if g and g.islower() and '`' not in g)
        stats['Verb SSTs']['nPred'] += sum(1 for p in ptags if p and p.islower() and '`' not in p)
        stats['Verb SSTs']['tp'] += sum(1 for g,p in zip(gtags,ptags) if p and p.islower() and '`' not in p and g==p)
        
        # preposition supersense labels
        stats['Prep SSTs']['nGold'] += sum(1 for g in gtags if g and not g.islower() and not g.isupper() and '`' not in g)
        stats['Prep SSTs']['nPred'] += sum(1 for p in ptags if p and not p.islower() and not p.isupper() and '`' not in p)
        stats['Prep SSTs']['tp'] += sum(1 for g,p in zip(gtags,ptags) if p and not p.islower() and not p.isupper() and '`' not in p and g==p)
        
        # only `a
        stats['Auxes']['nGold'] += sum(1 for g in gtags if g=='`a')
        stats['Auxes']['nPred'] += sum(1 for p in ptags if p=='`a')
        stats['Auxes']['tp'] += sum(1 for g,p in zip(gtags,ptags) if p=='`a' and g==p)
        
        
        # collapse all supersense labels together
        stats['Any SST']['nGold'] += sum(1 for g in gtags if g and '`' not in g)
        stats['Any SST']['nPred'] += sum(1 for p in ptags if p and '`' not in p)
        stats['Any SST']['tp'] += sum(1 for g,p in zip(gtags,ptags) if p and g and '`' not in p and '`' not in g)
        
        for g,p in zip(gtags,ptags):
            conf[g,p] += 1
            if g:
                stats[g]['nGold'] += 1
                stats[sstpos(g)]['nGold'] += 1
            if p:
                stats[p]['nPred'] += 1
                stats[sstpos(p)]['nPred'] += 1
                if g==p:
                    stats[p]['tp'] += 1
                    stats[sstpos(p)]['tp'] += 1
        
    stats['Exact Tag']['Acc'] = Ratio(stats['Exact Tag']['tp'], stats['Exact Tag']['nGold'])
    for x in stats:
        if x!='Exact Tag':
            stats[x]['P'] = Ratio(stats[x]['tp'], stats[x]['nPred'])
            stats[x]['R'] = Ratio(stats[x]['tp'], stats[x]['nGold'])
            stats[x]['F'] = f1(stats[x]['P'], stats[x]['R'])
    
    print(stats)
    print(conf)
    
    print('  Acc  |   P   |   R   |   F   || R: NSST | VSST |  `a  | PSST')
    parts = [(' {:.2%}'.format(float(stats['Exact Tag']['Acc'])),
              '{:>7}'.format(stats['Exact Tag']['Acc'].numeratorS),
              '{:>7}'.format(stats['Exact Tag']['Acc'].denominatorS))]
    parts += [(' {:.2%}'.format(float(stats['All Classes'][x])),
               '{:>7}'.format(stats['All Classes'][x].numeratorS),
               '{:>7}'.format(stats['All Classes'][x].denominatorS)) for x in ('P', 'R')]
    parts += [(' {:.2%}  '.format(float(stats['All Classes']['F'])),
               '         ',
               '         ')]
    parts += [(' {:.2%}'.format(float(stats[y]['R'])),
               '{:>7}'.format(stats[y]['R'].numeratorS),
               '{:>7}'.format(stats[y]['R'].denominatorS)) for y in ('Noun SSTs', 'Verb SSTs', 'Auxes', 'Prep SSTs')]
    for pp in zip(*parts):
        print(' '.join(pp))
