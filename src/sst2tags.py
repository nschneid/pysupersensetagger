#!/usr/bin/env python2.7
#coding=utf-8
'''
Reads an .mwe file with gold annotations, one sentence per line, 
and produces a .tags file (one word per line) using the 
OoBbĪĨīĩ encoding. If the sentence cannot be represented in this encoding, 
some groupings will be removed and a message written to stderr.

Input format (3 columns):

sentID   annotated_sentence   {"words": [[word1,pos1],...], "_": [[offset1,offset2],...], "~": [[offset1,offset2,offset3],...]}

Output is in the tab-separated format:

offset   word   lemma   POS   tag   parent   strength   label   sentId

- lemma will (for now) be left empty

@see: dataFeaturizer.SupersenseTrainSet

@author: Nathan Schneider (nschneid@cs.cmu.edu)
@since: 2014-06-04
'''
from __future__ import print_function, division
import os, sys, re, fileinput, codecs, json

I_BAR, I_TILDE, i_BAR, i_TILDE = 'ĪĨīĩ'.decode('utf-8')

def convert(inF, outF=sys.stdout):
    
    for ln in inF:
        sentId, anno, data = ln.strip().split('\t')
        data = json.loads(data)
        parents = {}
        gapstrength = {}    # offset -> kind of gap ('_' or '~'), if the offset lies with a gap
        
        # process strong groups
        for grp in data["_"]:
            g = sorted(grp)
            skip = False
            for i,j in zip(g[:-1],g[1:]):
                if j>i+1:
                    if i in gapstrength:    # gap within a gap
                        print('Simplifying: removing gappy group that is wholly contained within another gap:', g, anno, file=sys.stderr)
                        skip = True
                        break
            if skip: continue
            
            for i,j in zip(g[:-1],g[1:]):
                assert j not in parents
                parents[j] = i, '_'
                if j>i+1:
                    for h in range(i+1,j):
                        gapstrength[h] = '_'
        
        # process weak groups, skipping any that interleave with (are only partially contained in a gap of) 
        # a strong group
        for grp in data["~"]:
            g = sorted(grp)
            skip = False
            for i in g:
                if i in gapstrength and any(j for j in g if j not in gapstrength):
                    print('Simplifying: removing weak group that interleaves with a strong gap:', g, anno, file=sys.stderr)
                    skip = True
                    break
            if skip: continue
            for i,j in zip(g[:-1],g[1:]):
                if j>i+1:
                    if i in gapstrength:    # gap within a gap
                        print('Simplifying: removing gappy group that is wholly contained within another gap:', g, anno, file=sys.stderr)
                        skip = True
                        break
            if skip: continue
            
            for i,j in zip(g[:-1],g[1:]):
                if j not in parents:
                    parents[j] = i, '~'
                else:
                    assert parents[j][0]==i,(j,parents[j],i,g,anno)
                if j>i+1:
                    for h in range(i+1,j):
                        gapstrength.setdefault(h,'~')
        
        allparents = set(zip(*parents.values())[0]) if parents else set()
        
        labels = {int(k): l for k,(w,l) in data["labels"].items()}
        for i,(w,pos) in enumerate(data["words"]):
            parent, strength = parents.get(i+1,(0,''))
            label = labels.get(i+1,'')
            labelFlag = '-'+label if label else ''
            amInGap = (i+1 in gapstrength)
            if parent==0:
                if i+1 in allparents:
                    tag = ('b' if amInGap else 'B')+labelFlag
                else:
                    tag = ('o' if amInGap else 'O')+labelFlag
            elif strength=='_': # do not attach label to strong MWE continuations
                tag = i_BAR if amInGap else I_BAR
            else:
                assert strength=='~'
                tag = (i_TILDE if amInGap else I_TILDE)+labelFlag
            
            print(i+1, w, '', pos, tag.encode('utf-8'), parent, strength, label, sentId, sep='\t', file=outF)
        print()

if __name__=='__main__':
    convert(fileinput.input())
