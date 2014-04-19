# coding=UTF-8
'''
Handles the 'integrated' file format in the Penn Arabic Treebank. Written with reference to Part 3, v. 3.2.

Currently handles word-level information (tags, etc.) but not parses.

@author: Nathan Schneider (nschneid)
@since: 2010-06-22
'''

# Strive towards Python 3 compatibility
from __future__ import print_function, unicode_literals, division, absolute_import
from future_builtins import map, filter

import re, codecs, collections, os.path
import edu.cmu.cs.lti.ark.pyutil.fileutils as futil

SourceToken = collections.namedtuple('SourceToken', 'num ar bw chstart chend ttknstart ttknend a b c d')
TreeToken = collections.namedtuple('TreeToken', 'num tag f g voc gloss chstart chend ar bw y')
Sentence = collections.namedtuple('Sentence', 'num sourceTokens treeTokens')

documents = {}

if os.path.isdir('/mal2/corpora'):
    PATB_3_DIR = '/mal2/corpora/arabic_tb_3_3.2/atb3_v3_2'
else:
    PATB_3_DIR = 'edu/cmu/cs/lti/aqmar/data/ATB3-v3.2-sample'

class PATBDoc(object):
    def __init__(self, f):
        self.fileid = None
        self.sentences = []
        
        sourceTokens = []
        treeTokens = [] # after clitic separation
        sentid = None
        for ln in f:
            m = re.match('^CHUNK:(ANN\d{8}.\d{4}):(\d+)$', ln)  # e.g.: CHUNK:ANN20020115.0001:12
            if m is not None:
                if self.fileid is not None:
                    self.sentences.append(Sentence(sentid, sourceTokens, treeTokens))
                    sourceTokens = []
                    treeTokens = []
                    assert self.fileid==m.group(1)
                self.fileid = m.group(1)
                sentid = int(m.group(2)) # 1-based
            elif sentid is not None:
                if ln.startswith('s:'): # Source token (before clitic separation)
                    # e.g.: s:30 ·مسؤولون·ms&wlwn·180·188·34·34·1·[maso&uwl_1]·(maso&uwluwna)·OK
                    #m = re.match('^s:(\d+)\s+·(?P<ar>[^·]+)·(?P<bw>[^·]+)·\d+·\d+·(?P<ttknstart>\d+)·(?P<ttknend>\d+)·.*')
                    s = SourceToken(*tuple(re.split(ur'\s*·', ln[ln.index(':')+1:])))
                    
                    # convert numeric values to ints
                    sd = s._asdict()
                    for f in ('num','chstart','chend','ttknstart','ttknend','a'):
                        sd[f] = int(sd[f])
                    s = SourceToken(**sd)
                    
                    sourceTokens.append(s)
                elif ln.startswith('t:'):   # Tree token (after clitic separation)
                    # e.g.: t:34 ·NOUN+NSUFF_MASC_PL_NOM·f·f·maso&uwl+uwna·official/functionary + [masc.pl.]·180·188·مسؤولون·ms&wlwn·[]
                    t = TreeToken(*tuple(re.split(ur'\s*·', ln[ln.index(':')+1:])))
                    
                    # convert numeric values to ints
                    td = t._asdict()
                    for f in ('num','chstart','chend'):
                        td[f] = int(td[f])
                    t = TreeToken(**td)
                    
                    treeTokens.append(t)
                    
    def sent(self, sentence, romanized=True, separateClitics=True):
        tkns = sentence.treeTokens if separateClitics else sentence.sourceTokens
        chtype = 'bw' if romanized else 'ar'
        return [getattr(tkn,chtype) for tkn in tkns]
    
    def sents(self, romanized=True, separateClitics=True):
        return [self.sent(s, romanized=romanized, separateClitics=separateClitics) for s in self.sentences]
    
    def tagged_sent(self, sentence, romanized=True, separateClitics=True):
        tkns = sentence.treeTokens #if separateClitics else sentence.sourceTokens
        chtype = 'bw' if romanized else 'ar'
        tt = [(getattr(tkn,chtype), tkn.tag, tkn.voc, tkn.gloss) for tkn in tkns]
        if separateClitics:
            return tt
        
        # Join together tree tokens corresponding to each source word
        result = []
        for stkn in sentence.sourceTokens:
            w = tt[stkn.ttknstart:stkn.ttknend+1]
            result.append(tuple('++'.join(map(lambda x: x[i], w)) for i in range(4)))    # Separate properties with ++
        return result
    
    def tagged_sents(self, romanized=True, separateClitics=True):
        return [self.tagged_sent(s, romanized=romanized, separateClitics=separateClitics) for s in self.sentences]

    def morph_sent(self, sentence):
        '''@return: Full analyses of words in the sentence. Each word is represented by a tuple 
        of the SourceToken for the word and a list of corresponding TreeTokens (there will 
        be multiple TreeTokens if the word contains a clitic).
        '''
        result = []
        for stkn in sentence.sourceTokens:
            ttkns = [sentence.treeTokens[i] for i in range(stkn.ttknstart,stkn.ttknend+1)]
            result.append((stkn,ttkns))
        return result
    
    def morph_sents(self):
        return [self.morph_sent(s) for s in self.sentences]
    
    def voc_sent(self, sentence, segmented=True):
        '''@return: The vocalized and (optionally) segmented words of the sentence, in Buckwalter transliteration. 
        If 'segmented' is True, clitics will be separated by '++' and other morphemes by '+'.
        '''
        s = self.morph_sent(sentence)
        voc = ['++'.join(ttkn.voc for ttkn in w[1]) for w in s]
        if not segmented:
            return [w.replace('+','') for w in voc]
        return voc
    
    def voc_sents(self, segmented=True):
        return [self.voc_sent(s, segmented=segmented) for s in self.sentences]
    

def fileids():
    return list(map(lambda filename: filename[:-4], 
               futil.list_files(PATB_3_DIR + '/data/integrated', r'.*[.]txt$', recursive=False, absolute=False)))

def getDocPath(fileid):
    return PATB_3_DIR + '/data/integrated/{0}.txt'.format(fileid)

def docs(file_ids=None):
    fids = fileids() if file_ids is None else file_ids
    if isinstance(fids,basestring):
        return docs([fids])
    
    dd = []
    for fid in fids:
        if fid in documents:
            dd.append(documents[fid])
        else:   # load the file
            with codecs.open(getDocPath(fid),'rb','utf-8') as f:
                d = PATBDoc(f)
            documents[d.fileid] = d
            dd.append(d)
    return dd

def sents(file_ids=None, romanized=True, separateClitics=True):
    dd = docs(file_ids)        
    return reduce(lambda l1,l2: l1+l2, [d.sents(romanized=romanized, separateClitics=separateClitics) for d in dd])

def tagged_sents(file_ids=None, romanized=True, separateClitics=True):
    dd = docs(file_ids)
    return reduce(lambda l1,l2: l1+l2, [d.tagged_sents(romanized=romanized, separateClitics=separateClitics) for d in dd])

def morph_sents(file_ids=None):
    '''@see: PATBDoc.morph_sent()'''
    dd = docs(file_ids)
    return reduce(lambda l1,l2: l1+l2, [d.morph_sents() for d in dd])

def voc_sents(self, file_ids=None, segmented=True):
    '''@see: PATBDoc.voc_sent()'''
    dd = docs(file_ids)
    return reduce(lambda l1,l2: l1+l2, [d.voc_sents(segmented=segmented) for d in dd])

def test():
    print(sents('ANN20020115.0001'))
    print(tagged_sents('ANN20020115.0001', separateClitics=False))
    print(sents())

if __name__=='__main__':
    test()
