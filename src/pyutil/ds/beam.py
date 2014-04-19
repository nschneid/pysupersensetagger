'''
Beam class: holds up to N key-value pairs, 
prioritizing higher values.

@author: Nathan Schneider (nschneid)
'''

from collections import Counter, defaultdict

class Beam(Counter):
    '''
    >>> b = Beam(3)
    >>> b['x'] = 9
    >>> b['a'] = 3
    >>> b['y'] = 5
    >>> b
    Beam(3, {'x': 9, 'y': 5, 'a': 3})
    >>> b['q'] = 7
    >>> b
    Beam(3, {'x': 9, 'q': 7, 'y': 5})
    >>> del b['y']
    >>> b
    Beam(3, {'x': 9, 'q': 7})
    >>> b['g'] = -1
    >>> b
    Beam(3, {'x': 9, 'q': 7, 'g': -1})
    >>> Beam(3, {'a': 3, 'x': 9, 'q': 7, 'y': 5, 'g': -1})
    Beam(3, {'x': 9, 'q': 7, 'y': 5})
    >>> Beam(3, Counter('aaxaxxxqqqxxxqqqyyyxxqyyr'))
    Beam(3, {'x': 9, 'q': 7, 'y': 5})
    '''
    def __init__(self, n, *args, **kwargs):
        assert n>=1
        self._size = n
        self._worst = None
        Counter.__init__(self, *args, **kwargs)
        if len(self)>=n:
            self._worst = self.most_common(n)[-1]
            for k,v in self.most_common()[n:]:
                del self[k]
    
    def __setitem__(self, k, v):
        if k in self:
            assert self[k]==v
            return
        if self._worst is not None: # the beam is full
            if v<=self._worst[1]:
                return
            else:
                del self[self._worst[0]]
                
        Counter.__setitem__(self, k, v)
        assert self._size>=len(self)
        if self._size==len(self):
            self._worst = min(self.items(), key=lambda (k1,v1): v1)
    
    def __delitem__(self, k):
        if self._worst and self._worst[0]==k:
            self._worst = None
        Counter.__delitem__(self, k)
        
    def __repr__(self):
        s = Counter.__repr__(self)
        return s[:s.index('(')+1] + '{}, '.format(self._size) + s[s.index('(')+1:]

if __name__=='__main__':
    import doctest
    doctest.testmod()
