'''
Created on Jul 19, 2013

@author: Nathan Schneider (nschneid)
'''

class Trie(object):
    '''
    Trie (prefix tree) data structure for mapping sequences to values.
    Values can be overridden, but removal from the trie is not currently supported.
    
    >>> t = Trie()
    >>> t['panther'] = 'PANTHER'
    >>> t['panda'] = 'PANDA'
    >>> t['pancake'] = 'PANCAKE'
    >>> t['pastrami'] = 'PASTRAMI'
    >>> t['pastafarian'] = 'PASTAFARIAN'
    >>> t['noodles'] = 'NOODLES'
        
    >>> for s in ['panther', 'panda', 'pancake', 'pastrami', 'pastafarian', 'noodles']:
    ...    assert s in t
    ...    assert t.get(s)==s.upper()
    >>> 'pescatarian' in t
    False
    >>> 'pastafarian' in t
    True
    >>> 'pasta' in t
    False
    >>> print(t.get('pescatarian'))
    None
    >>> t.longest('pasta', False)
    False
    >>> t.longest('pastafarian')
    (('p', 'a', 's', 't', 'a', 'f', 'a', 'r', 'i', 'a', 'n'), 'PASTAFARIAN')
    >>> t.longest('pastafarianism')
    (('p', 'a', 's', 't', 'a', 'f', 'a', 'r', 'i', 'a', 'n'), 'PASTAFARIAN')
    
    >>> t[(3, 1, 4)] = '314'
    >>> t[(3, 1, 4, 1, 5, 9)] = '314159'
    >>> t[(0, 0, 3, 1, 4)] = '00314'
    >>> t.longest((3, 1, 4))
    ((3, 1, 4), '314')
    >>> (3, 1, 4, 1, 5) in t
    False
    >>> print(t.get((3, 1, 4, 1, 5)))
    None
    >>> t.longest((3, 1, 4, 1, 5))
    ((3, 1, 4), '314')
    '''
    def __init__(self):
        self._map = {}  # map from sequence items to embedded Tries
        self._vals = {} # map from items ending a sequence to their values
    
    def __setitem__(self, seq, v):
        first, rest = seq[0], seq[1:]
        if rest:
            self._map.setdefault(first, Trie())[rest] = v
        else:
            self._vals[first] = v
    
    def __contains__(self, seq):
        '''@return: whether a value is stored for 'seq' '''
        first, rest = seq[0], seq[1:]
        if rest:
            if first not in self._map:
                return False
            return rest in self._map[first]
        return first in self._vals
    
    def get(self, seq, default=None):
        '''@return: value associated with 'seq' if 'seq' is in the trie, 'default' otherwise'''
        first, rest = seq[0], seq[1:]
        if rest:
            if first not in self._map:
                return default
            return self._map[first].get(rest)
        else:
            return self._vals.get(first, default)
        
    def longest(self, seq, default=None):
        '''@return: pair of longest prefix of 'seq' 
        corresponding to a value in the Trie, and that value. 
        If no such prefix of 'seq' has a value, returns 'default'.'''
        
        first, rest = seq[0], seq[1:]
        longer = self._map[first].longest(rest, default) if rest and first in self._map else default
        if longer==default: # 'rest' is empty, or none of the prefix of 'rest' leads to a value
            if first in self._vals:
                return ((first,), self._vals[first])
            else:
                return default
        else:
            return ((first,)+longer[0], longer[1])

if __name__=='__main__':
    import doctest
    doctest.testmod()
