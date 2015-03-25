#!/usr/bin/env python2.7
#coding=utf-8
'''
Reads a .tags file (one word per line using the OoBbĪĨīĩ encoding)
and produces a .sst file with grouped MWE offsets, one sentence per line,
generating a human-readable annotation of the segmentation 
and including labels, lemmas (if available), and tags in a JSON object.

Input is in the tab-separated format:

offset   word   lemma   POS   tag   parent   strength   label   sentId

Output format (3 columns):

sentID   annotated_sentence   {"words": [[word1,pos1],...], "labels": {"offset1": [word1,label1], "offset2": [word2,label2]}, "lemmas": [lemma1,lemma2,...], "tags": [tag1,tag2,...], "_": [[offset1,offset2],...], "~": [[offset1,offset2,offset3],...]}

With the -l flag, show labels in annotated_sentence. 
Otherwise, annotated_sentence will only contain the segmentation.

@author: Nathan Schneider (nschneid@cs.cmu.edu)
@since: 2014-06-07
'''
from __future__ import print_function, division
import os, sys, re, fileinput, codecs, json
from __builtin__ import True

I_BAR, I_TILDE, i_BAR, i_TILDE = 'ĪĨīĩ'.decode('utf-8')

def render(ww, sgroups, wgroups, labels={}):
    '''
    Converts the given lexical annotation to a string 
    with _ and ~ as weak and strong joiners, respectively.
    Assumes this can be done straightforwardly (no nested gaps, 
    no weak expressions involving words both inside and outside 
    of a strong gap, no weak expression that contains only 
    part of a strong expression, etc.). 
    Also does not specially escape of tokens containing _ or ~.
    
    Note that indices are 1-based.
    
    >>> ww = ['a','b','c','d','e','f']
    >>> render(ww, [[2,3],[5,6]], [[1,2,3,5,6]])
    u'a~b_c~ d ~e_f'
    >>> render(ww, [], [], {3: 'C', 6: 'FFF'})
    u'a b c|C d e f|FFF'
    >>> render(ww, [[2,3],[5,6]], [], {2: 'BC', 5: 'EF'})
    u'a b_c|BC d e_f|EF'
    >>> render(ww, [[1,2,6],[3,4,5]], [], {1: 'ABF'})
    u'a_b_ c_d_e _f|ABF'
    >>> render(ww, [[1,2,6],[3,4,5]], [], {1: 'ABF', 3: 'CDE'})
    u'a_b_ c_d_e|CDE _f|ABF'
    >>> render(ww, [], [[3,4,5]], {4: 'D', 5: 'E', 6: 'F'})
    u'a b c~d|D~e|E f|F'
    >>> render(ww, [], [[3,5]])
    u'a b c~ d ~e f'
    >>> render(ww, [[2,3],[5,6]], [[2,3,4]], {4: 'D'})
    u'a b_c~d|D e_f'
    >>> render(ww, [[2,3],[5,6]], [[1,2,3,5,6]])
    u'a~b_c~ d ~e_f'
    >>> render(ww, [[2,3],[5,6]], [[1,2,3,4,5,6]], {1: 'A', 2: 'BC', 4: 'D', 5: 'EF'})
    u'a|A~b_c|BC~d|D~e_f|EF'
    >>> render(ww, [[2,4],[5,6]], [[2,4,5,6]], {2: 'BD', 3: 'C'})
    u'a b_ c|C _d|BD~e_f'
    '''
    singletonlabels = dict(labels)  # will be winnowed down to the labels not covered by a strong MWE
    before = [None]*len(ww)   # None by default; remaining None's will be converted to empty strings
    labelafter = ['']*len(ww)
    after = [None]*len(ww)   # None by default; remaining None's will be converted to spaces
    for group in sgroups:
        g = sorted(group)
        for i,j in zip(g[:-1],g[1:]):
            if j==i+1:
                after[i-1] = ''
                before[j-1] = '_'
            else:
                after[i-1] = '_'
                before[i] = ' '
                before[j-1] = '_'
                after[j-2] = ' '
        if g[0] in labels:
            labelafter[g[-1]-1] = '|'+labels[g[0]]
            del singletonlabels[g[0]]
    for i,lbl in singletonlabels.items():
        assert i-1 not in labelafter
        labelafter[i-1] = '|'+lbl
    for group in wgroups:
        g = sorted(group)
        for i,j in zip(g[:-1],g[1:]):
            if j==i+1:
                if after[i-1] is None and before[j-1] is None:
                    after[i-1] = ''
                    before[j-1] = '~'
            else:
                if after[i-1] is None and before[i] is None:
                    after[i-1] = '~'
                    before[i] = ' '
                if after[j-2] is None and before[j-1] is None:
                    before[j-1] = '~'
                    after[j-2] = ' '
    
    after = ['' if x is None else x for x in after]
    before = [' ' if x is None else x for x in before]
    return u''.join(sum(zip(before,ww,labelafter,after), ())).strip()

def process_sentence(words, lemmas, tags, labels, parents, sentId=None):
    # form groups
    sgroups = []
    wgroups = []
    i2sgroup = {}
    i2wgroup = {}
    for offset,(parent,strength) in sorted(parents.items()):
        if strength in {'_',''}:
            if parent not in i2sgroup:
                i2sgroup[parent] = len(sgroups)
                sgroups.append([parent])
            i2sgroup[offset] = i2sgroup[parent]
            sgroups[i2sgroup[parent]].append(offset)
    for offset,(parent,strength) in sorted(parents.items()):
        if strength=='~':   # includes transitive closure over all member strong groups
            if parent not in i2wgroup:
                i2wgroup[parent] = len(wgroups)
                wgroups.append([])
            i2wgroup[offset] = i2wgroup[parent]
            g = wgroups[i2wgroup[offset]]
            
            if parent in i2sgroup: # include strong group of parent
                for o in sgroups[i2sgroup[parent]]:
                    if o not in g:  # avoid redundancy if weak group has 3 parts
                        g.append(o)
            elif parent not in g:
                g.append(parent)
            
            if offset in i2sgroup:  # include strong group of child
                for o in sgroups[i2sgroup[offset]]:
                    i2wgroup[o] = i2wgroup[offset]  # in case the last word in a strong expression precedes part of a weak expression
                    g.append(o)
            else:
                g.append(offset)
    
    # sanity check: number of tokens belonging to some MWE
    assert len(set(sum(sgroups+wgroups,[])))==sum(1 for t in tags if t[0].upper()!='O'),(tags,sgroups,wgroups)
    
    # sanity check: no token shared by multiple strong or multiple weak groups
    assert len(set(sum(sgroups,[])))==len(sum(sgroups,[])),(sgroups,tags,sentId)
    assert len(set(sum(wgroups,[])))==len(sum(wgroups,[])),(wgroups,tags,sentId)
    
    data = {"words": words, "tags": tags, "_": sgroups, "~": wgroups,
            "labels": {k+1: [words[k][0],lbl] for k,lbl in enumerate(labels) if lbl}}
    if any(lemmas):
        data["lemmas"] = lemmas
    
    return data

def convert(inF, outF=sys.stdout, labelsInRenderedAnno=False):
    def readsent():
        words = []
        lemmas = []
        tags = []
        labels = []
        parents = {}
        
        for ln in inF:
            if not ln.strip():
                if not words: continue
                break
                    
            parts = ln[:-1].split('\t')
            if len(parts)==9:
                offset, word, lemma, POS, tag, parent, strength, label, sentId = parts
            else:
                sentId = ''
                offset, word, lemma, POS, tag, parent, strength, label = parts
            words.append((word, POS))
            lemmas.append(lemma)
            tags.append(tag)
            labels.append(label)
            if int(parent)!=0:
                parents[int(offset)] = (int(parent), strength)
        
        if not words: raise StopIteration()
        
        data = process_sentence(words, lemmas, tags, labels, parents, sentId=sentId)
        
        print(sentId, 
              render(zip(*words)[0], data["_"], data["~"], 
                     ({int(k): v[1] for k,v in data["labels"].items()} if labelsInRenderedAnno else {})), 
              json.dumps(data), sep='\t', file=outF)
    
    while True:
        try:
            readsent()
        except StopIteration:
            break
        

if __name__=='__main__':
    args = sys.argv[1:]
    if args and args[0]=='-l':
        labelsInRenderedAnno = True
        args = args[1:]
    else:
        labelsInRenderedAnno = False
    convert(fileinput.input(args), labelsInRenderedAnno=labelsInRenderedAnno)
    #import doctest
    #doctest.testmod()

