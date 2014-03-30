'''
Created on Jun 2, 2013

@author: Nathan Schneider (nschneid)
'''
from __future__ import print_function, division

import os, sys, codecs

from labeledSentence import LabeledSentence
import morph

USTRINGS = {}
def uintern(unicode_string):
    '''Simulate built-in intern(), but in a way that works for unicode strings.'''
    return USTRINGS.setdefault(unicode_string,unicode_string)

class DataSet(object):
    def __init__(self, f):
        self._file = f
    def __iter__(self):
        return 

class SupersenseDataSet(DataSet):
    def __init__(self, path, labels, legacy0, require_gold=True, keep_in_memory=True):
        self._path = path
        self._labels = labels
        self._cache = [] if keep_in_memory else None
        self._legacy0 = legacy0
        self._require_gold = require_gold
        self.open_file()
    
    def close_file(self):
        self._f.close()
    
    def open_file(self):
        self._f = codecs.open(self._path, 'r', 'utf-8')
    
    def reset(self):
        '''Stop reading more input instances, and prepare to start at the beginning.'''
        if self._cache is None:
            self._f.seek(0)
        else:
            self.close_file()
        self._reset = True  # TODO: not sure about this mechanism for short-circuiting existing iterators
    
    def __iter__(self, autoreset=True):
        '''
        Load the BIO tagged supersense data from Semcor, as provided in 
        the SuperSenseTagger release (SEM_07.BI).
        We also use their POS labels, which presumably were what their 
        paper used.
        One difference is that this method expects the data to be converted 
        into a 3-column format with an extra newline between each sentence 
        (as in CoNLL data), which can be created from the SST data with 
        a short perl script. 
        '''
        self._reset = False
        if self._f.closed:
            assert self._cache
            for sent in self._cache:
                yield sent
        else:
            sent = LabeledSentence()
            for ln in self._f:
                if not ln.strip():
                    if len(sent)>0:
                        if self._cache is not None:
                            self._cache.append(sent)
                        yield sent
                        if self._reset:
                            raise StopIteration()
                        sent = LabeledSentence()
                    continue
                parts = ln[:-1].split('\t')
                
                if len(parts)>3:
                    if parts[3]!='':
                        sent.sentId = parts[3]
                    parts = parts[:3]
                if not self._require_gold:
                    token, pos = parts[:2]
                    label = parts[2] if len(parts)>2 and parts[2].strip() else None
                else:
                    token, pos, label = parts
                
                if label is not None:
                    if label=='0' and self._legacy0:
                        assert 'O' in self._labels,self._labels
                        label = 'O'
                    elif label not in self._labels:
                        label = 'O'
                    label = uintern(unicode(label))
                    
                pos = uintern(unicode(pos))
                stemS = uintern(unicode(morph.stem(token,pos)))
                sent.addToken(token=token, stem=stemS, pos=pos, goldLabel=label)
                
            if len(sent)>0:
                if self._cache is not None:
                    self._cache.append(sent)
                yield sent
                
            if autoreset:
                self.reset()


class SupersenseFeaturizer(object):
    
    def __init__(self, extractor, dataset, indexes, cache_features=True):
        self._extractor = extractor
        self._data = dataset
        self._featureIndexes = indexes
        self._features = [] if cache_features else None
        # TODO: For now, try caching just the lifted 0-order features.
    
    def __iter__(self):
        for j,sent in enumerate(self._data):
            if self._features is None or j>=len(self._features):  # not yet in cache
                lexiconCandidatesThisSent = self._extractor.extractLexiconCandidates(sent)
                
                o0FeatsEachToken = []
                
                for i in range(len(sent)):
                    # zero-order features (lifted)
                    o0FeatureMap = self._extractor.extractFeatureValues(sent, i, 
                                                                                   usePredictedLabels=False, 
                                                                                   orders={0}, 
                                                                                   indexer=self._featureIndexes, 
                                                                                   candidatesThisSentence=lexiconCandidatesThisSent)
                    
                    if not o0FeatureMap:
                        raise Exception('No 0-order features found for this token')
                    
                    o0FeatsEachToken.append(o0FeatureMap)
                    
                if self._features is not None:
                    assert len(self._features)==j
                    self._features.append(o0FeatsEachToken)
                
                yield sent,o0FeatsEachToken
            else:
                yield sent,self._features[j]
    
    def enable_caching(self):
        if self._features is None:
            self._features = []
    
    def reset(self):
        self._data.reset()
