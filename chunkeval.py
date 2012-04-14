#!/usr/bin/env python2.7
'''
Several measures for evaluating chunking predictions with respect to a gold standard.
Pass -h to see usage.

Terminology used below:
 * 'token': an instance of an atomic unit, such as a word;
 * 'tag': indicates grouping and (possibly) classification of a token, e.g. B-PERSON
 * 'mention': a sequence of tokens according to some tagging;
 * 'label': indicates classification of a mention (all tokens in the mention 
   must agree for a given tagging)
 * 'position marker': the tag minus the label (if present), e.g. B
 * 'continuation marker': a position marker indicating the second or subsequent 
   position within a mention, e.g. I with BIO tagging
 * 'mention boundaries': the sequence of position markers associated with a mention
 * 'entity': a mention type according to a given tagging; label may or 
   may not be included as part of the type

The first mandatory option is the position marker scheme, which must be one of:
 * IO - inside/outside
 * BIO - beginning/inside/outside (the sequence O I is illegal)
 * BILOU (Ratinov & Roth, 2009) - beginning/inside/last/outside/unique 
   (illegal sequences: O {I,L}; B O; L I; {B,I} U; U {I,L})

Every tag must start with the position marker. It may optionally be followed 
by a hyphen and a label. Tokens unlabeled in the input are implicitly given 
a special "null" label, except for O, which must not be labeled.
(Unlabeled scores will therefore work if the gold standard has labels 
but the predictions do not, or vice versa.) The label for a continuation tag 
must match the label of the previous tag.

The input format follows the CoNLL conventions: each line consists of a 
token, a gold tag, and a predicted tag, separated by tabs. Blank lines 
indicate mandatory chunk boundaries (e.g. sentence boundaries if tagging 
words).

The character ` has a special meaning when used in the first and/or second 
tag field: it causes the entire token to be ignored when reading in the file. 
This can be used to abstain from judging predictions for certain tokens, though 
it should be used with caution as it may result in a different chunk interpretation 
of neighboring tokens. E.g., the sequences

a    O    B-1
b    `    B-2
c    O    I-2

and

a    O    O
b    `    B
c    O    I

will both be rendered invalid if the middle token is ignored. If ` is used for all 
tokens in a sequence, the sequence will safely be ignored.

@since: 2012-01-30
@author: Nathan Schneider (nschneid)
'''

from __future__ import print_function, division, absolute_import

import codecs, sys, os, re, fileinput, argparse
from collections import Counter, defaultdict

if __name__ == "__main__" and __package__ is None:
    import scoring
else:
    from . import scoring


IGNORE_SYMBOL = '`'

def isContinuation(tag, scheme='BIO'):
    pm = tag[0]
    if pm in 'BOU':
        return False
    if pm=='I':
        return 'B' in scheme
    if pm=='L':
        return True

def isPrimary(tag, scheme='BIO'):
    '''
    The *primary* tag is the one in a privileged position within the 
    mention, a position occurring only once per mention:
      * If the scheme contains B, B tags are primary
      * If the scheme contains U, U tags are primary
      * If the scheme contains L but not B, L tags are primary
    '''
    pm = tag[0]
    if pm in 'BU': return True
    if pm=='L' and 'B' not in scheme: return True
    return False

def primarize(positionMarker, scheme='BIO'):
    if isPrimary(positionMarker, scheme): return positionMarker
    if 'B' in scheme: return 'B'
    if 'L' in scheme: return 'L'
    assert False

def tokenConfusions(goldseq, predseq, ignoreLabels=False, collapseNonO=False, scheme='BIO', bag=False, ignoreContinuation=False):
    n = nFound = nMissed = nExtra = 0
    
    if bag:
        gC = Counter()
        pC = Counter()
        for g,p in zip(goldseq,predseq):
            gpm, gl = g
            ppm, pl = p
            
            if ignoreLabels:
                gl = pl = None
            
            if ignoreContinuation:
                if collapseNonO:
                    if gpm!='O': gpm = 'B' if isPrimary(gpm,scheme) else gpm
                    if ppm!='O': ppm = 'B' if isPrimary(ppm,scheme) else ppm
                
                if gpm=='O' or isPrimary(gpm,scheme):
                    gC[(gpm,gl)] += 1
                if ppm=='O' or isPrimary(ppm,scheme):
                    pC[(ppm,pl)] += 1
            else:
                if collapseNonO:
                    if gpm!='O': gpm = primarize(gpm,scheme)
                    if ppm!='O': ppm = primarize(ppm,scheme)
                
                gC[(gpm,gl)] += 1
                pC[(ppm,pl)] += 1
            
        n = sum(gC.values())
        assert ignoreContinuation or n==sum(pC.values())

        for tag in set(gC.keys()+pC.keys()):
            if tag[0]=='O': continue
            gn, pn = gC[tag], pC[tag]
            nFound += min(gn,pn)
            if gn>pn:
                nMissed += gn-pn
            elif gn<pn:
                nExtra += pn-gn
    else:
        for g,p in zip(goldseq,predseq):
            gpm, gl = g
            ppm, pl = p
            n += 1
            if (gpm==ppm or (collapseNonO and (gpm=='O')==(ppm=='O'))) and (ignoreLabels or gl==pl):    # correct tag
                if gpm!='O':
                    nFound += 1 # true positive
            elif ppm=='O':
                nMissed += 1    # false negative
            else:
                nExtra += 1 # false positive
    return scoring.ConfusionMatrix(Both=nFound, Aonly=nMissed, Bonly=nExtra, Neither=n-nFound-nMissed-nExtra)

def mentionSpans(seq, includeOTokens=False, value='full' or 'label', scheme='BIO'):
    '''
    >>> d = mentionSpans([('B', 'PER'), ('I', 'PER'), ('I', 'PER'), ('B', 'ORG'), ('O', None), ('O', None), ('B', 'LOC')], includeOTokens=False)
    >>> assert d=={(0, 3): [('B', 'PER'), ('I', 'PER'), ('I', 'PER')], (3, 4): [('B', 'ORG')], (6, 7): [('B', 'LOC')]}, d
    >>> d = mentionSpans([('B', 'PER'), ('I', 'PER'), ('I', 'PER'), ('B', 'ORG'), ('O', None), ('O', None), ('B', 'LOC')], includeOTokens=True)
    >>> assert d=={(0, 3): [('B', 'PER'), ('I', 'PER'), ('I', 'PER')], (3, 4): [('B', 'ORG')], (4, 5): [('O', None)], (5, 6): [('O', None)], (6, 7): [('B', 'LOC')]}, d
    >>> d = mentionSpans([('B', 'PER'), ('I', 'PER'), ('I', 'PER'), ('B', 'ORG'), ('O', None), ('O', None), ('B', 'LOC')], includeOTokens=False, value='label')
    >>> assert d=={(0, 3): 'PER', (3, 4): 'ORG', (6, 7): 'LOC'}, d
    >>> d = mentionSpans([('B', 'PER'), ('I', 'PER'), ('I', 'PER'), ('B', 'ORG'), ('O', None), ('O', None), ('B', 'LOC')], includeOTokens=True, value='label')
    >>> assert d=={(0, 3): 'PER', (3, 4): 'ORG', (4, 5): None, (5, 6): None, (6, 7): 'LOC'}, d
    
    '''
    spanMap = {}
    i = 0
    j = 1
    while j<len(seq)+1:
        if j==len(seq) or not isContinuation(seq[j][0], scheme):
            if j-i>1 or (includeOTokens or seq[i][0]!='O'):
                spanMap[(i,j)] = seq[i:j]
            i = j
        j += 1
        
    assert value in ('full', 'label')
    if value=='full':
        return spanMap
    return {k: v[0][1] for k,v in spanMap.items()}


def overlap(span1, span2):
    a, b = span1
    i, j = span2
    assert a<b and i<j
    return a<=i<b or i<=a<j

def softMentionConfusions(goldseq, predseq, ignoreLabels=False, matchCriterion=overlap, scheme='BIO'):
    '''
    Any partial overlap between a gold and predicted mention counts as a match between the two mentions.
    Ignores labels.
    
    True positives and true negatives don't really make sense here, so we return 0 counts for these 
    and use precision vs. recall calculations.
    
    >>> gold = [('B', 'PER'), ('I', 'PER'), ('I', 'PER'), ('B', 'ORG'), ('O', None), ('O', None), ('B', 'LOC'), ('I', 'LOC')]
    >>> pred = [('B', 'PER'), ('O', None), ('B', 'ORG'), ('I', 'ORG'), ('O', None), ('B', 'YYY'), ('B', 'ORG'), ('B', 'XXX')]
    >>> conf, precRatio, recRatio = softMentionConfusions(gold, pred, ignoreLabels=True)
    >>> assert conf==scoring.ConfusionMatrix(0, 1, 0, 0), conf
    >>> assert precRatio==Counter(numer=4, denom=5)
    >>> assert recRatio==Counter(numer=3, denom=3)
    
    >>> gold = [('B', 'PER'), ('I', 'PER'), ('I', 'PER'), ('B', 'ORG'), ('O', None), ('O', None), ('B', 'LOC'), ('I', 'LOC')]
    >>> pred = [('B', 'PER'), ('O', None), ('B', 'ORG'), ('I', 'ORG'), ('O', None), ('B', 'YYY'), ('B', 'ORG'), ('B', 'XXX')]
    >>> conf, precRatio, recRatio = softMentionConfusions(gold, pred, ignoreLabels=False)
    >>> assert conf==scoring.ConfusionMatrix(1, 3, 0, 0), conf
    >>> assert precRatio==Counter(numer=2, denom=5)
    >>> assert recRatio==Counter(numer=2, denom=3)
    '''
    x = dict.keys if ignoreLabels else dict.items
    goldMentionSpans = set(x(mentionSpans(goldseq, includeOTokens=False, value='label', scheme=scheme)))
    goldOSpans = set(x(mentionSpans(goldseq, includeOTokens=True, value='label', scheme=scheme))).difference(goldMentionSpans)
    predMentionSpans = set(x(mentionSpans(predseq, includeOTokens=False, value='label', scheme=scheme)))
    predOSpans = set(x(mentionSpans(predseq, includeOTokens=True, value='label', scheme=scheme))).difference(predMentionSpans)
    
    if ignoreLabels:
        match = lambda g,p: matchCriterion(g,p)
    else:
        match = lambda g,p: matchCriterion(g[0],p[0]) and g[1]==p[1]
    
    nMatchedPred = nExtra = 0
    uncoveredGold = set(goldMentionSpans)
    for p in predMentionSpans:
        matchedGold = {p} if p in goldMentionSpans else {g for g in goldMentionSpans if match(g,p)}
        if matchedGold:
            nMatchedPred += 1
            uncoveredGold.difference_update(matchedGold)
        else:   # prediction doesn't overlap with any gold mention
            nExtra += 1
    nMatchedGold = len(goldMentionSpans) - len(uncoveredGold)
    return (scoring.ConfusionMatrix(Aonly=len(uncoveredGold), Bonly=nExtra, Both=0, Neither=0), 
            Counter(numer=nMatchedPred, denom=len(predMentionSpans)),
            Counter(numer=nMatchedGold, denom=len(goldMentionSpans)))

def mentionConfusions(goldseq, predseq, ignoreLabels=False, scheme='BIO'):
    '''
    >>> gold = [('B', 'PER'), ('I', 'PER'), ('I', 'PER'), ('B', 'ORG'), ('O', None), ('O', None), ('B', 'LOC'), ('B', 'XXX')]
    >>> pred = [('B', 'PER'), ('O', None), ('B', 'ORG'), ('I', 'ORG'), ('O', None), ('B', 'YYY'), ('B', 'ORG'), ('B', 'XXX')]
    >>> conf = mentionConfusions(gold, pred, ignoreLabels=True)
    >>> assert conf==scoring.ConfusionMatrix(2, 3, 2, 1), conf
    
    >>> gold = [('B', 'PER'), ('I', 'PER'), ('I', 'PER'), ('B', 'ORG'), ('O', None), ('O', None), ('B', 'LOC'), ('B', 'XXX')]
    >>> pred = [('B', 'PER'), ('O', None), ('B', 'ORG'), ('I', 'ORG'), ('O', None), ('B', 'YYY'), ('B', 'ORG'), ('B', 'XXX')]
    >>> conf = mentionConfusions(gold, pred, ignoreLabels=False)
    >>> assert conf==scoring.ConfusionMatrix(3, 4, 1, 1), conf
    '''
    x = dict.keys if ignoreLabels else dict.items
    goldMentionSpans = set(x(mentionSpans(goldseq, includeOTokens=False, value='label', scheme=scheme)))
    goldOSpans = set(x(mentionSpans(goldseq, includeOTokens=True, value='label', scheme=scheme))).difference(goldMentionSpans)
    predMentionSpans = set(x(mentionSpans(predseq, includeOTokens=False, value='label', scheme=scheme)))
    predOSpans = set(x(mentionSpans(predseq, includeOTokens=True, value='label', scheme=scheme))).difference(predMentionSpans)
    return scoring.ConfusionMatrix(len(goldMentionSpans.difference(predMentionSpans)), len(predMentionSpans.difference(goldMentionSpans)), 
                                   len(goldMentionSpans & predMentionSpans), len(goldOSpans & predOSpans))


def manningChunks(goldseq, predseq, scheme='BIO'):    # TODO: right now this assumes BIO. IO and BILOU may require changes.
    '''
    Provided tag sequences must consist of pairs of the form (position marker, label).
    
    Categorize errors in the provided tagging according to the scheme proposed by Chris Manning in 
      http://nlpers.blogspot.com/2006/08/doing-named-entity-recognition-dont.html
    which consists of breaking sequences based on the combination of gold and predicted taggings 
    and assigning each chunk to one of seven categories: 
      tn, tp, fn, fp, le (label error), be (boundary error), lbe (label and boundary error)
    
    His bracketing criteria: "Moving along the sequence, the subsequence boundaries are: 
     (i) at start and end of document, 
     (ii) anywhere there is a change to or from a word/O/O token from or to a token 
     where either guess or gold is not O, and 
     (iii) anywhere that both systems change their class assignment [I interpret this as any transition 
     out of and/or into a mention. -NSS] simultaneously, regardless of whether they agree."
    
    The returned sequence is a list of tuples, where each tuple combines the tags 
    of a gold subsequence, the tags of a predicted sequence, and one of the seven 
    error categories.
    
    >>> manningChunks([('O', None), ('B', 'PERS'), ('I', 'PERS'), ('O', None)],
    ...               [('B', 'PERS'), ('I', 'PERS'), ('B', 'PERS'), ('O', None)])
    [([('O', None), ('B', 'PERS'), ('I', 'PERS')], [('B', 'PERS'), ('I', 'PERS'), ('B', 'PERS')], 'be'), ([('O', None)], [('O', None)], 'tn')]
    
    >>> tests = []
    
    # Manning's examples don't explicitly include B or I tags, but we assume 
    # they contain no two consecutive B tags with the same label.
    
    >>> tests.append(('drove/O/O along/O/O a/O/O narrow/O/O road/O/O', 'tn'))
    >>> tests.append(('in/O/O Palo/LOC/LOC Alto/LOC/LOC ./O/O', 'tn,tp,tn'))
    >>> tests.append(('in/O/O Palo/LOC/O Alto/LOC/O ./O/O', 'tn,fn,tn'))
    >>> tests.append(('an/O/O Awful/O/ORG Headache/O/ORG ./O/O', 'tn,fp,tn'))
    >>> tests.append(('I/O/O live/O/O in/O/O Palo/LOC/ORG Alto/LOC/ORG ./O/O', 'tn,le,tn'))
    >>> tests.append(('Unless/O/PERS Karl/PERS/PERS Smith/PERS/PERS resigns/O/O', 'be,tn'))
    >>> tests.append(('Unless/O/ORG Karl/PERS/ORG Smith/PERS/ORG resigns/O/O', 'lbe,tn'))
    >>> for seq,cats in tests:
    ...    tkns,goldseq,predseq = zip(*(itm.split('/') for itm in seq.split()))
    ...    goldseqBIO = [('O',None) if l2=='O' else (('I',l2) if l2==l1 else ('B',l2)) for l1,l2 in zip((None,)+goldseq, goldseq)]
    ...    predseqBIO = [('O',None) if l2=='O' else (('I',l2) if l2==l1 else ('B',l2)) for l1,l2 in zip((None,)+predseq, predseq)]
    ...    chks = manningChunks(goldseqBIO, predseqBIO)
    ...    assert [chk[2] for chk in chks]==cats.split(',')
    
    
    >>> tests = []
    
    # Additional examples
    
    >>> tests.append(([('O', None), ('O', None), ('O', None), ('B', 'LOC'), ('I', 'LOC'), ('O', None)],
    ...               [('O', None), ('O', None), ('O', None), ('B', 'ORG'), ('B', 'ORG'), ('O', None)], 'tn,lbe,tn'))
    >>> tests.append(([('O', None), ('O', None), ('O', None), ('B', 'LOC'), ('B', 'LOC'), ('O', None)],
    ...               [('O', None), ('O', None), ('O', None), ('B', 'ORG'), ('I', 'ORG'), ('O', None)], 'tn,lbe,tn'))
    >>> tests.append(([('O', None), ('B', 'PERS'), ('I', 'PERS'), ('O', None)],
    ...               [('B', 'PERS'), ('I', 'PERS'), ('B', 'PERS'), ('O', None)], 'be,tn'))
    >>> tests.append(([('O', None), ('B', 'PERS'), ('I', 'PERS'), ('O', None)],
    ...               [('B', 'PERS'), ('B', 'PERS'), ('I', 'PERS'), ('O', None)], 'fp,tp,tn'))
    >>> tests.append(([('O', None), ('B', 'PERS'), ('I', 'PERS'), ('O', None)],
    ...               [('B', 'PERS'), ('B', 'PERS'), ('B', 'PERS'), ('O', None)], 'fp,be,tn'))
    >>> tests.append(([('O', None), ('B', 'PERS'), ('B', 'PERS'), ('O', None)],
    ...               [('B', 'PERS'), ('B', 'PERS'), ('B', 'ORG'), ('O', None)], 'fp,tp,le,tn'))
    >>> tests.append(([('O', None), ('B', 'PERS'), ('I', 'PERS'), ('O', None)],
    ...               [('B', 'ORG'), ('B', 'ORG'), ('I', 'ORG'), ('O', None)], 'fp,le,tn'))
    >>> tests.append(([('O', None), ('B', 'PERS'), ('I', 'PERS'), ('O', None), ('O', None), ('B', 'XXX')],
    ...               [('B', 'ORG'), ('B', 'ORG'), ('I', 'ORG'), ('O', None), ('O', None), ('B', 'XXX')], 'fp,le,tn,tp'))
    >>> tests.append(([('O', None), ('B', 'PERS'), ('I', 'PERS'), ('O', None), ('B', 'XXX'), ('B', 'XXX')],
    ...               [('B', 'ORG'), ('B', 'ORG'), ('I', 'ORG'), ('O', None), ('O', None), ('B', 'XXX')], 'fp,le,tn,fn,tp'))
    >>> tests.append(([('O', None), ('B', 'PERS'), ('I', 'PERS'), ('B', 'XXX'), ('I', 'XXX'), ('B', 'XXX')],
    ...               [('B', 'ORG'), ('B', 'ORG'), ('I', 'ORG'), ('O', None), ('O', None), ('B', 'XXX')], 'fp,le,fn,tp'))
    >>> tests.append(([('O', None), ('B', 'PERS'), ('I', 'PERS'), ('B', 'PERS'), ('I', 'PERS'), ('B', 'PERS')],
    ...               [('O', None), ('B', 'PERS'), ('I', 'PERS'), ('O', None), ('O', None), ('B', 'PERS')], 'tn,tp,fn,tp'))
    >>> tests.append(([('O', None), ('B', 'PERS'), ('I', 'PERS'), ('B', 'PERS'), ('I', 'PERS'), ('B', 'XXX')],
    ...               [('O', None), ('B', 'PERS'), ('I', 'PERS'), ('O', None), ('O', None), ('B', 'PERS')], 'tn,tp,fn,le'))
    >>> tests.append(([('O', None), ('B', 'PERS'), ('I', 'PERS'), ('B', 'PERS'), ('I', 'PERS'), ('B', 'XXX')],
    ...               [('O', None), ('B', 'PERS'), ('I', 'PERS'), ('O', None), ('B', 'PERS'), ('I', 'PERS')], 'tn,tp,lbe'))
    
    >>> for goldseqBIO,predseqBIO,cats in tests:
    ...    chks = manningChunks(goldseqBIO, predseqBIO)
    ...    assert [chk[2] for chk in chks]==cats.split(','),(goldseqBIO,predseqBIO,cats,chks)
    '''
    
    
    def nextChunk(goldseq, predseq, i):
        gg = []
        pp = []
        j = i
        while j<len(goldseq):
            g = goldseq[j]
            p = predseq[j]
            if len(gg)>0:
                if (gg[-1][0]=='O' and pp[-1][0]=='O')!=(g[0]=='O' and p[0]=='O'):
                    break   # transition to or from word/O/O
                elif (not isContinuation(g[0], scheme) and (g[0]!='O' or gg[-1][0]!='O')) and (not isContinuation(p[0], scheme) and (p[0]!='O' or pp[-1][0]!='O')):
                    break   # transition out of and/or into a mention: i.e. any tag not continuing a mention or series of O's
            gg.append(g)
            pp.append(p)
            j += 1
        
        assert len(gg)==len(pp)
        
        # boundary strings
        gpmS = ''.join(g[0] for g in gg)
        ppmS = ''.join(p[0] for p in pp)
        
        # error type
        if gpmS==ppmS:    # boundaries agree
            if {g[1] for g in gg if g[0]!='O'}=={p[1] for p in pp if p[0]!='O'}:  # labels agree
                cat = 'tn' if set(gpmS)=={'O'} else 'tp'
            else:
                cat = 'le'
        else:   # boundary error
            if set(gpmS)=={'O'}:
                cat = 'fp'
            elif set(ppmS)=={'O'}:
                cat = 'fn'
            elif {g[1] for g in gg if g[0]!='O'}=={p[1] for p in pp if p[0]!='O'}:  # labels agree
                cat = 'be'
            else:
                cat = 'lbe'
            
        return (gg, pp, cat)
    
    assert len(goldseq)==len(predseq)
    i = 0
    chunks = []
    while i<len(goldseq):
        chk = nextChunk(goldseq, predseq, i)
        chunks.append(chk)
        i += len(chk[0])
    assert i==len(goldseq)
    
    return chunks
    
def manningCounts(goldseq, predseq, scheme='BIO'):
    chunks = manningChunks(goldseq, predseq, scheme=scheme)
    return Counter(chk[2] for chk in chunks for i in range(len(chk[0]))), Counter(chk[2] for chk in chunks)
    
def manningScore(goldseq, predseq, scheme='BIO'):
    '''
    One scoring scheme based on Manning chunks (see above) 
    that is intended to combat the traditional mention F score's
    bias against proposing entities.
    
    @return: "Precision" and "recall" values per this approach
    
    >>> gold = [('B', 'PER'), ('I', 'PER'), ('I', 'PER'), ('B', 'ORG'), ('O', None), ('O', None), ('B', 'LOC'), ('B', 'XXX')]
    >>> pred = [('B', 'PER'), ('O', None), ('B', 'ORG'), ('I', 'ORG'), ('O', None), ('B', 'YYY'), ('B', 'ORG'), ('B', 'XXX')]
    >>> manningScore(gold, pred)    # (5-1-1-.5)/5, (4-1-0-.5)/4
    (0.5, 0.625)
    
    >>> gold = [('B', 'PER'), ('I', 'PER'), ('I', 'PER'), ('O', None), ('O', None), ('O', None), ('B', 'LOC'), ('B', 'XXX')]
    >>> pred = [('B', 'PER'), ('O', None), ('B', 'ORG'), ('I', 'ORG'), ('O', None), ('B', 'YYY'), ('B', 'ORG'), ('B', 'XXX')]
    >>> manningScore(gold, pred)==(.5, 2/3)    # (5-1-1-.5)/5, (3-.5-0-.5)/3
    True
    '''
    chunks = manningChunks(goldseq, predseq, scheme=scheme)
    precDemerits = 0
    recDemerits = 0
    for gg,pp,errcat in chunks:
        if errcat=='fp':
            precDemerits += 1
        elif errcat=='fn':
            recDemerits += 1
        elif errcat in ('be','le','lbe'):
            recDemerits += 0.5*len(mentionSpans(gg, includeOTokens=False, scheme=scheme))
            precDemerits += 0.5*len(mentionSpans(pp, includeOTokens=False, scheme=scheme))
    nGold = len(mentionSpans(goldseq, includeOTokens=False, scheme=scheme))
    nGuesses = len(mentionSpans(predseq, includeOTokens=False, scheme=scheme))
    return (nGuesses-precDemerits)/nGuesses, (nGold-recDemerits)/nGold
    
def ensureSequence(seq, scheme='BIO' or 'IO' or 'BILOU', fixProblems=False):
    '''
    Check that the tag sequence is legal under the current tagging scheme. 
    Raise an informative error if it is invalid.
    '''
    seq = list(seq)
    pmS = ''.join(t[0] for t in seq)   # position markers
    pmsNotInScheme = set(pmS).difference(set(scheme))
    if pmsNotInScheme:
        raise Exception('One or more position markers not allowed by the {} tagging scheme: {}'.format(scheme, pmsNotInScheme))
    
    if 'B' in scheme and (pmS[0]=='I' or pmS[0]=='L'):
        s = 'Illegal position marker at the beginning of a sequence (tagging scheme {}): {}'.format(scheme, pmS[0])
        if fixProblems:
            print(s, file=sys.stderr)
            seq[0] = ('B', seq[0][1])
        else:
            raise Exception(s)
    if 'L' in scheme and pmS[-1]=='I':
        s = 'Illegal position marker at the end of a sequence (tagging scheme {}): {}'.format(scheme, pmS[-1])
        if fixProblems:
            print(s, file=sys.stderr)
            seq[-1] = ('L', seq[-1][1])
        else:
            raise Exception(s)
    if 'B' in scheme and ('OI' in pmS or 'UI' in pmS or 'LI' in pmS):
        s = 'Illegal position marker sequence (tagging scheme {}): O I or U I or L I (I must always continue a mention)'.format(scheme)
        if fixProblems:
            print(s, file=sys.stderr)
            m = re.search(r'[OUL]I', pmS)
            while m:
                i = m.start()
                seq[i] = ('B', seq[i+1][1])
                pmS = ''.join(t[0] for t in seq)
                m = re.search(r'[OUL]I', pmS)
        else:
            raise Exception(s)
    if 'OL' in pmS or 'UL' in pmS:
        raise Exception('Illegal position marker sequence (tagging scheme {}): O L or U L (L must always continue a mention)'.format(scheme))
    if 'U' in scheme and ('BB' in pmS or 'BO' in pmS or 'BU' in pmS or 'IU' in pmS or 'LL' in pmS or 'OL' in pmS):
        raise Exception('Illegal position marker sequence (tagging scheme {}): B B or B O or B U or I U or L L or O L (must use U for all and only length-1 mentions)'.format(scheme))
    
    # ensure each mention has a consistent label
    for t1,t2 in zip(seq, seq[1:]):
        pm2, l2 = t2
        if pm2=='O' and l2 is not None:
            raise Exception('Illegal label for an O tag: {}'.format(l2))
        if isContinuation(pm2, scheme=scheme) and l2!=t1[1]:
            raise Exception("Continuation tag's label ({}) is inconsistent with previous label ({})".format(l2, t1[1]))
        
    return seq

def slashFormat(tkns,golds,preds):
    return u' '.join(u'{}/{}/{}'.format(t, u'-'.join(g) if g[1] is not None else g[0], u'-'.join(p) if p[1] is not None else p[0]) for t,g,p in zip(tkns,golds,preds))

def loadSequences(conllF, scheme='BIO'):
    '''
    Generator over sequences in the input file, where each sequence is a list of token triples of the form
      (token_word, (gold_position_marker, gold_label), (pred_position_marker, pred_label))
    An error is raised if any of the sequences are ill-formed.
    '''
    
    
    def nextSequence(conllF, scheme='BIO'):
        '''@return: The next (non-ignored) sequence as a list, or [] if there is no remaining sequence.'''
        seq = []
        nIgnored = 0
        for ln in conllF:
            ln = ln[:-1]
            if ln.strip()=='':
                if seq: break
                continue
            tkn, goldt, predt = ln.split('\t')
            assert tkn,'Missing token on line: {}'.format(ln)
            assert goldt,'Missing first (gold) tag on line: {}'.format(ln)
            assert predt,'Missing second (predicted) tag on line: {}'.format(ln)
            
            if goldt==IGNORE_SYMBOL or predt==IGNORE_SYMBOL:
                nIgnored += 1
                continue
            
            gold = goldt.split('-') if '-' in goldt else (goldt, None)
            gold = (str(gold[0]), gold[1])  # convert from unicode
            pred = tuple(predt.split('-')) if '-' in predt else (predt, None)
            pred = (str(pred[0]), pred[1])
            assert len(gold)==len(pred)==2
            seq.append((tkn, gold, pred))
            
        if nIgnored>0:
            global nIgnoredTokens, nIgnoredSeqs
            nIgnoredTokens += nIgnored
            nIgnoredSeqs += 1   # this sequence was at least partially ignored
            if not seq: # this sequence was entirely ignored; get the next one
                return nextSequence(conllF, scheme)
            
        return seq
    
    while True:
        seq = nextSequence(conllF, scheme=scheme)
        if not seq:
            break
        tkns,golds,preds = zip(*seq)
        try:
            golds = ensureSequence(golds, scheme=scheme, fixProblems=True)
            preds = ensureSequence(preds, scheme=scheme, fixProblems=True)
        except:
            print('Ending at line {}:'.format(fileinput.lineno()), slashFormat(tkns,golds,preds).encode('utf-8'), file=sys.stderr)
            raise
        yield zip(tkns,golds,preds)
        

#import doctest
#doctest.testmod()

'''
Error report:

              00000  TOKENS  00000               00000  MENTIONS  00000
        found xtra miss O/O                found xtra miss O/O
          tp   fp   fn    tn   P% R% F1%     tp   fp   fn    tn   P% R% F1%
Exact L  0000 0000 0000 000000 00 00 00.0   0000 0000 0000 000000 00 00 00.0
Exact UL 0000 0000 0000 000000 00 00 00.0   0000 0000 0000 000000 00 00 00.0    # ignoring labels
Soft L   0000 0000 0000 000000 00 00 00.0   0000 0000 0000 000000 00 00 00.0    # ignoring position markers / mention match if at least one of its tokens is found with the correct label
Soft UL  0000 0000 0000 000000 00 00 00.0   0000 0000 0000 000000 00 00 00.0    # unlabeled: collapsing B & I / mention match if at least one of its tokens is found
Bag L    0000 0000 0000 000000 00 00 00.0   0000 0000 0000 000000 00 00 00.0    # treating each sentence as a bag of labeled items
Bag UL   0000 0000 0000 000000 00 00 00.0   0000 0000 0000 000000 00 00 00.0    # treating each sentence as a bag of unlabeled items
Manning  0000 0000 0000 000000              0000 0000 0000 000000           
         le: 0000  be: 0000  lbe: 0000      le: 0000  be: 0000  lbe: 0000   
'''

if __name__=='__main__':
    #print(scoring.ConfusionMatrix(Aonly=4, Bonly=0, Both=2, Neither=8).asPRF())
    
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', action='store_true', help='Display a legend explaining the output')
    parser.add_argument('-p', action='store_true', help='Show percentages instead of counts for tp, fp, etc.')  # TODO:
    parser.add_argument('-n', action='store_true', help='Show ratios instead of percentages for precision and recall')  # TODO:
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-l', metavar='LABEL', action='append', help='Specify one of the labels to select (tokens corresponding to unselected labels will be counted as O)')  # TODO
    group.add_argument('-L', action='store_true', help='Suppress per-label reports')
    parser.add_argument('-m', metavar='THRESHOLD', action='store', type=lambda v: float(v) if '.' in v or 'e' in v else int(v),
                        help='Specify a minimum threshold for values to display for tp, fp, etc. '
                        'This can be a count (an integer) or a percentage (between 0 and 1). 0 entries will not display; '
                        'nonzero entries below the threshold will display as *.')    # TODO:
    parser.add_argument('scheme', choices=['BIO','IO','BILOU'], help='Tagging scheme')
    parser.add_argument('conllFiles', nargs='*')
    args = parser.parse_args()
    
    scheme = args.scheme
    
    
    def newConfsMap():
        confs = {}
        for x in ['Exact L', 'Exact UL', 'Soft L', 'Soft UL', 'Bag L', 'Bag UL', 'Manning']:
            confs[x] = {'token': scoring.ConfusionMatrix(0,0,0,0), 'mention': scoring.ConfusionMatrix(0,0,0,0)}
        confs['Manning2'] = {'token': Counter(), 'mention': Counter()}
        softPR = {'UL': {'P': Counter(numer=0, denom=0), 'R': Counter(numer=0, denom=0)}, 
                   'L':  {'P': Counter(numer=0, denom=0), 'R': Counter(numer=0, denom=0)}}
        return confs, softPR
    
    data = defaultdict(newConfsMap)   # labelset => confs map
    
    
    nSeqs = Counter()   # label set => number of sequences having some (predicted or gold) label in the set
    nTokens = Counter() # label set => number of tokens in the sequences corresponding to this label set
    allLabels = set()
    
    global nIgnoredTokens, nIgnoredSeqs
    nIgnoredTokens = 0
    nIgnoredSeqs = 0
    
    sys.stdin = codecs.getreader("utf-8")(sys.stdin)
    for seq in loadSequences(fileinput.input(args.conllFiles, openhook=fileinput.hook_encoded("utf-8")), scheme):
        tkns,golds,preds = zip(*seq)
        tkns,golds,preds = list(tkns),list(golds),list(preds)
        labelsThisSeq = set(itm[1] for itm in golds+preds if itm[0]!='O')
        allLabels.update(labelsThisSeq)
        
        selectedLbls = args.l
        if selectedLbls:
            lblsets = {tuple(selectedLbls)} # a specific subset of labels
        elif args.L:
            lblsets = {()}  # all labels
        else:
            lblsets = {(lbl,) for lbl in allLabels} | {()}  # all labels, plus each label individually
        
        for lblset in lblsets:
            if lblset==('LOC',):
                pass
            
            if lblset!=() and not set(lblset)&labelsThisSeq:
                continue
            
            nSeqs[lblset] += 1
            nTokens[lblset] += len(tkns)
            if lblset!=():
                goldseq = [((pm,lbl) if lbl in lblset+(None,) else ('O',None)) for pm,lbl in golds]
                predseq = [((pm,lbl) if lbl in lblset+(None,) else ('O',None)) for pm,lbl in preds]
                # lblset+(None,) because if e.g. the predictions are unlabeled (their label is stored as None), 
                # we want to be able to compute unlabeled scores even if the gold tokens are labeled
            else:
                goldseq = golds
                predseq = preds
            
            confs, softPR = data[lblset]
            confs['Exact UL']['token'] += tokenConfusions(goldseq, predseq, ignoreLabels=True, collapseNonO=False, scheme=scheme)
            confs['Exact L']['token'] += tokenConfusions(goldseq, predseq, ignoreLabels=False, collapseNonO=False, scheme=scheme)
            confs['Soft UL']['token'] += tokenConfusions(goldseq, predseq, ignoreLabels=True, collapseNonO=True, scheme=scheme)
            confs['Soft L']['token'] += tokenConfusions(goldseq, predseq, ignoreLabels=False, collapseNonO=True, scheme=scheme)
            confs['Bag UL']['token'] += tokenConfusions(goldseq, predseq, ignoreLabels=True, collapseNonO=True, scheme=scheme, bag=True)
            confs['Bag L']['token'] += tokenConfusions(goldseq, predseq, ignoreLabels=False, collapseNonO=True, scheme=scheme, bag=True)
            confs['Exact UL']['mention'] += mentionConfusions(goldseq, predseq, ignoreLabels=True, scheme=scheme)
            confs['Exact L']['mention'] += mentionConfusions(goldseq, predseq, ignoreLabels=False, scheme=scheme)
            softConf, softP, softR = softMentionConfusions(goldseq, predseq, ignoreLabels=True, scheme=scheme)
            confs['Soft UL']['mention'] += softConf
            softPR['UL']['P'] += softP
            softPR['UL']['R'] += softR
            softConf, softP, softR = softMentionConfusions(goldseq, predseq, ignoreLabels=False, scheme=scheme)
            confs['Soft L']['mention'] += softConf
            softPR['L']['P'] += softP
            softPR['L']['R'] += softR
            confs['Bag UL']['mention'] += tokenConfusions(goldseq, predseq, ignoreLabels=True, collapseNonO=True, scheme=scheme, bag=True, ignoreContinuation=True)
            confs['Bag L']['mention'] += tokenConfusions(goldseq, predseq, ignoreLabels=False, collapseNonO=True, scheme=scheme, bag=True, ignoreContinuation=True)
            manningTkn, manningChk = manningCounts(goldseq, predseq, scheme)
            confs['Manning2']['token'] += manningTkn
            confs['Manning2']['mention'] += manningChk
            
    if len(data)==0:
        print('No relevant sequences for the given options', file=sys.stderr)
        
    if nIgnoredTokens>0:
        print('Ignoring {} tokens in {} sequences'.format(nIgnoredTokens, nIgnoredSeqs), file=sys.stderr)
    
    for lblset,(confs,softPR) in sorted(data.items(), key=lambda itm: itm[0]):
        if lblset==():
            lblsS = 'All {} labels'.format(len(allLabels))
        else:
            if len(allLabels)==1:
                continue
            lblsS = 'Labels: '+' '.join('(null)' if lbl is None else (repr(lbl) if re.search(r'\s|[\'"]', lbl) else lbl) for lbl in lblset)
            unseen = set(lblset).difference(allLabels)
            if unseen:
                print('Warning: some labels never seen in data:', unseen, file=sys.stderr)
        
        c = confs['Manning2']['token']
        confs['Manning']['token'] = scoring.ConfusionMatrix(Both=c['tp'], Neither=c['tn'], Aonly=c['fn'], Bonly=c['fp'])
        c = confs['Manning2']['mention']
        confs['Manning']['mention'] = scoring.ConfusionMatrix(Both=c['tp'], Neither=c['tn'], Aonly=c['fn'], Bonly=c['fp'])
        
        nGoldMentions = confs['Exact UL']['mention'].Atotal
        nPredMentions = confs['Exact UL']['mention'].Btotal
        
        print('''
{}
    {:5}             {:5}  TOKENS  {:<5}           {:5}  MENTIONS  {:<5}
        found xtra miss O/O                found xtra miss O/O
          tp   fp   fn    tn   P% R% F1%     tp   fp   fn    tn   P% R% F1%'''.format(lblsS, scheme, nTokens[lblset], nSeqs[lblset], nGoldMentions, nPredMentions))
        
        # TODO: Manning score?
        for x in ['Exact L', 'Exact UL', 'Soft L', 'Soft UL', 'Bag L', 'Bag UL', 'Manning', 'Manning2']:
            print('{:8} '.format(x if x!='Manning2' else ''), end='')
            
            # token-level info
            conf = confs[x]['token']
            if x=='Manning2':
                print('le: {:4}  be: {:4}  lbe: {:4}      '.format(conf['le'], conf['be'], conf['lbe']), end='')
            else:
                print('{:4} {:4} {:4} {:6} '.format(conf.Both, conf.Bonly, conf.Aonly, conf.Neither), end='')
                #print(conf)
                prf = conf.asPRF(suppressZeroDenominatorCheck=True)
                
                if x!='Manning':
                    print('{: >2.0f} {: >2.0f} {: >4.1f}   '.format(100*prf.P, 100*prf.R, 100*prf.F), end='')
                else:
                    print('{:2} {:2} {:4}   '.format('','',''), end='')
    
            # mention-level info
            conf = confs[x]['mention']
            if x=='Manning2':
                print('le: {:4}  be: {:4}  lbe: {:4}      '.format(conf['le'], conf['be'], conf['lbe']))
            else:
                print('{:>4} {:4} {:4} {:>6} '.format(conf.Both if 'Soft' not in x else '-', conf.Bonly, conf.Aonly, conf.Neither if 'Soft' not in x else '-'), end='')
                if x.startswith('Soft'):
                    ratios = softPR[x.split()[1]]['P'], softPR[x.split()[1]]['R']
                    prec = 100*ratios[0]['numer']/ratios[0]['denom'] if ratios[0]['denom']>0 else float('NaN')
                    rec = 100*ratios[1]['numer']/ratios[1]['denom'] if ratios[1]['denom']>0 else float('NaN')
                    prf = [prec, rec, scoring.harmonicMean(prec, rec)]
                else:
                    hardPRF = conf.asPRF(suppressZeroDenominatorCheck=True)
                    prf = [100*hardPRF.P, 100*hardPRF.R, 100*hardPRF.F]
                    
                if x!='Manning':
                    print('{: >2.0f} {: >2.0f} {: >4.1f}'.format(*prf), end='')
                print()
    if args.v:
        print('''
---------------------------------------------------------------------------------
                                   L E G E N D
---------------------------------------------------------------------------------
(Selected labels)
(0)          (1)  TOKENS  (2)                   (3)  MENTIONS  (4)
        found xtra miss O/O                found xtra miss O/O
          tp   fp   fn    tn   P% R% F1%     tp   fp   fn    tn   P% R% F1%
Exact L  0000 0000 0000 000000 00 00 00.0   0000 0000 0000 000000 00 00 00.0  (7)
Exact UL 0000 0000 0000 000000 00 00 00.0   0000 0000 0000 000000 00 00 00.0  (8)
Soft L   0000 0000 0000 000000 00 00 00.0      - 0000 0000      - 00 00 00.0  (9)
Soft UL  0000 0000 0000 000000 00 00 00.0      - 0000 0000      - 00 00 00.0  (10)
Bag L    0000 0000 0000 000000 00 00 00.0   0000 0000 0000 000000 00 00 00.0  (11)
Bag UL   0000 0000 0000 000000 00 00 00.0   0000 0000 0000 000000 00 00 00.0  (12)
Manning  0000 0000 0000 000000              0000 0000 0000 000000             (13)
         le: 0000  be: 0000  lbe: 0000      le: 0000  be: 0000  lbe: 0000      |
         ------------- (5) --------------   ------------- (6) -------------- 

(Selected labels) Subset of labels used to calculate scores in this table. 
Any (gold or predicted) tag with a label not in this set will be replaced with O. 
(Tags without a label will be retained.)

(0) tagging scheme, e.g. BIO
(1) # tokens in sequences containing the selected labels at least once (gold or predicted)
(2) # sequences (sentences) containing the selected labels at least once (gold or predicted)
(3) # gold mentions
(4) # predicted mentions

(5) token-level, (6) mention-level statistics
COLUMNS true positives, false positives, false negatives, true negatives, 
        precision, recall, F1

MEASURES
(7) Exact, labeled: tokens/mentions only count if the prediction exactly matches the gold 
(8) Exact, unlabeled: category labels are ignored--only boundary errors are penalized
(9) Soft, labeled: token-level measures consider only the label in matching; 
      mention-level measures allow matches where any token is shared in common between mentions 
      on the two sides, provided those mentions have the same label
(10) Soft, unlabeled: token-level measures ignore the label and disregard the B vs. I distinction; 
      mention-level measures are computed as in (9) but ignoring the labels 
For (9) and (10), mention-level tp and tn measures are omitted because 1-many, many-1, and 
many-to-many alignments are possible.
(11) Bag, labeled: each sequence (sentence) is treated as a bag of tokens. Mention-level measures 
      consider exactly one token per mention. Positional differences are disregarded: all non-O 
      tags count the same for token-level measures and U is never distinguished from B in the 
      mention-level measures.
(12) Bag, unlabeled: each sequence is treated as a bag of tokens as in (11), but labels are ignored. 
(13) Manning: counts of error types based on Chris Manning's chunking scheme; token and chunk 
      (error event) counts are given for the results (no mention spans multiple chunks).
      le = label error, be = boundary error, lbe = label and boundary error; tp/tn reflect 
      exact matches and fp/fn indicate predictions that do not overlap with a gold mention.
''')
    # TODO:
    # - type-level measures?
    # - unordered (per-sequence bag) measures, including a binary version (e.g. "how many sentences had this label at least once")?
    # - test on real data
    # - counts of label confusions?
