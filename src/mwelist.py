#!/usr/bin/env python2.7
#coding=utf-8
'''
Reads an .mwe or .sst file with lemmas and produces a list of (lemmatized) MWE types
with the associated strength.

Input format (3 columns; additional fields may be present in the JSON object but will be ignored):

sentID   annotated_sentence   {"words": [[word1,pos1],...], "lemmas": [lemma1,lemma2,...], "_": [[offset1,offset2,offset3],...], "~": [[offset1,offset2,offset3],...], ...}

Options:
  -i: instead of counts, show each individual occurrence with its inflected token form, sentence ID, and first token offset

@author: Nathan Schneider (nschneid@cs.cmu.edu)
@since: 2014-06-04
'''
from __future__ import print_function, division
from collections import Counter, defaultdict
import os, sys, re, fileinput, codecs, json


def mwe_lemma_counts(inF, outF=sys.stdout, individual=False):
    result = Counter()
    for ln in inF:
        if not ln.strip(): continue
        sentId, anno, data = ln[:-1].split('\t')
        data = json.loads(data)
        for sg in data["_"]:
            key = (u' '.join(data["lemmas"][i-1] for i in sg),"_")
            if individual:
                key = key + (u' '.join(data["words"][i-1][0] for i in sg),sentId,sg[0])
            result[key] += 1
        for wg in data["~"]:
            key = key = (u' '.join(data["lemmas"][i-1] for i in wg),"~")
            if individual:
                key = key + (u' '.join(data["words"][i-1][0] for i in wg),sentId,wg[0])
            result[key] += 1
    return result

if __name__=='__main__':
    args = sys.argv[1:]
    if args and args[0]=='-i':
        individual = True
        args = args[1:]
    else:
        individual = False
        
    if individual:
        for (mwetype,strength,tokform,sentId,offset),n in sorted(mwe_lemma_counts(fileinput.input(args), individual=individual).items()):
            print(mwetype.encode('utf-8'), strength, tokform, sentId, offset, sep='\t')
    else:
        for (mwetype,strength),n in sorted(mwe_lemma_counts(fileinput.input(args), individual=individual).items()):
            print(mwetype.encode('utf-8'), strength, n, sep='\t')
