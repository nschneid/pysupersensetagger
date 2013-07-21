'''
Created on Jul 19, 2013

@author: Nathan Schneider (nschneid)
'''
from collections import Counter

class SequentialStringIndexer(object):
    '''Feature alphabet. Optional cutoff threshold; count is determined by the number of 
    calls to add() or setdefault() before calling freeze().'''
    def __init__(self, cutoff=None):
        self._s2i = {}
        self._i2s = []
        self._frozen = False
        self._cutoff = cutoff
        if self._cutoff is not None:
            self._counts = Counter()
    def setcount(self, k, n):
        '''Store a count for the associated entry. Useful for entries that should be 
        kept when thresholding but which might not be counted by calls to add()/setdefault().'''
        i = k if isinstance(k,int) else self._s2i[k]
        self._counts[i] = n
    def freeze(self):
        '''Locks the alphabet so it can no longer be modified and filters 
        according to the cutoff threshold (if specified).'''
        if self._cutoff is not None:  # apply feature cutoff (threshold)
            assert self._counts
            self._i2s = [s for i,s in enumerate(self._i2s) if self._counts[i]>=self._cutoff]
            if 0<=self._cutoff<=1: assert len(self._i2s)==len(self._s2i)
            del self._counts
            self._s2i = {s: i for i,s in enumerate(self._i2s)}
        self._frozen = True
        self._len = len(self._i2s)  # cache the length for efficiency
    def unfreeze(self):
        self._frozen = False
    def is_frozen(self):
        return self._frozen
    def __getitem__(self, k):
        if isinstance(k,int):
            return self._i2s[k]
        return self._s2i[k]
    def get(self, k, default=None):
        if isinstance(k,int):
            if k>=len(self._i2s):
                return default
            return self._i2s[k]
        return self._s2i.get(k,default)
    def __contains__(self, k):
        if isinstance(k,int):
            assert k>0
            return k<len(self._i2s)
        return k in self._s2i
    def add(self, s):
        if s not in self:
            if self.is_frozen():
                raise ValueError('Cannot add new item to frozen indexer: '+s)
            self._s2i[s] = i = len(self._i2s)
            self._i2s.append(s)
        elif not self.is_frozen() and self._cutoff is not None:
            i = self[s]
        if not self.is_frozen() and self._cutoff is not None:
            # update count
            self._counts[i] += 1
    def setdefault(self, k):
        '''looks up k, adding it if necessary'''
        self.add(k)
        return self[k]
    def __len__(self):
        return self._len if self.is_frozen() else len(self._i2s)
    @property
    def strings(self):
        return self._i2s
    def items(self):
        '''iterator over (index, string) pairs, sorted by index'''
        return enumerate(self._i2s)

class IndexedStringSet(set):
    '''Wraps a set contains indices to strings, with mapping provided in an indexer.
    Used to hold the features active for a particular instance.
    '''
    def __init__(self, indexer):
        self._indexer = indexer
        self._indices = set()
    @property
    def strings(self):
        return {self._indexer[i] for i in self._indices}
    @property
    def indices(self):
        return self._indices
    def __iter__(self):
        return iter(self._indices)
    def __len__(self):
        return len(self._indices)
    def add(self, k):
        '''If k is not already indexed, index it before adding it to the set'''
        if isinstance(k, int):
            assert k in self._indexer
            self._indices.add(k)
        else:
            self._indices.add(self._indexer.setdefault(k))
    def setdefault(self, k):
        self.add(k)
        return self._indexer[k]

class IndexedFeatureMap(object):
    '''The feature-value mapping for a particular instance.'''
    def __init__(self, indexer, default=1):
        self._set = IndexedStringSet(indexer)
        self._map = {}
        self._default = default
    def __setitem__(self, k, v):
        '''Add the specified feature/value pair if the feature is already indexed or 
        can be added to the index. Has no effect if the featureset is frozen and the 
        provided feature is not part of the featureset.'''
        if not self._set._indexer.is_frozen() or k in self._set._indexer:
            i = self._set.setdefault(k)
            if v!=self._default:
                self._map[i] = v
    def __iter__(self):
        return iter(self._set)
    def __len__(self):
        return len(self._set)
    def items(self):
        for i in self._set:
            yield (i, self._map.get(i, self._default))
    def named_items(self):
        for i in self._set:
            yield (self._set._indexer[i], self._map.get(i, self._default))
    def __repr__(self):
        return 'IndexedFeatureMap(['+', '.join(map(repr,self.named_items()))+'])'
