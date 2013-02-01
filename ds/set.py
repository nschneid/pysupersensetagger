'''
OrderedSet implementation, from http://stackoverflow.com/questions/1653970/does-python-have-an-ordered-set/1653978#1653978

The index() method and a few unit tests have been added.

@author: Nathan Schneider (nschneid)
@since: 2010-08-11
'''

# Strive towards Python 3 compatibility
from __future__ import print_function, unicode_literals, division, absolute_import
from future_builtins import map, filter

import collections

class OrderedSet(collections.OrderedDict, collections.MutableSet):
    '''
    A set that preserves the ordering of its entries.
    
    >>> {3,2,9,2}=={9,2,3}
    True
    >>> x = OrderedSet([3,2,9,2])
    >>> x == OrderedSet([2,9,3])
    False
    >>> x == OrderedSet([3,2,3,9,2])
    True
    >>> [y for y in x]
    [3, 2, 9]
    >>> x.index(2)
    1
    >>> x.index(0)
    Traceback (most recent call last):
      ...
    ValueError: 0 is not in set
    >>> [y for y in {3,2,9}]
    [9, 2, 3]
    '''
    
    def update(self, *args, **kwargs):
        if kwargs:
            raise TypeError("update() takes no keyword arguments")

        for s in args:
            for e in s:
                self.add(e)

    def add(self, elem):
        self[elem] = None

    def discard(self, elem):
        self.pop(elem, None)
        
    def index(self, elem):
        try:
            return self.keys().index(elem)
        except ValueError:
            raise ValueError('{} is not in set'.format(elem))
    
    def __le__(self, other):
        return all(e in other for e in self)

    def __lt__(self, other):
        return self <= other and self != other

    def __ge__(self, other):
        return all(e in self for e in other)

    def __gt__(self, other):
        return self >= other and self != other

    def __repr__(self):
        return 'OrderedSet([%s])' % (', '.join(map(repr, self.keys())))

    def __str__(self):
        return '{%s}' % (', '.join(map(repr, self.keys())))

    difference = property(lambda self: self.__sub__)
    difference_update = property(lambda self: self.__isub__)
    intersection = property(lambda self: self.__and__)
    intersection_update = property(lambda self: self.__iand__)
    issubset = property(lambda self: self.__le__)
    issuperset = property(lambda self: self.__ge__)
    symmetric_difference = property(lambda self: self.__xor__)
    symmetric_difference_update = property(lambda self: self.__ixor__)
    union = property(lambda self: self.__or__)

def test():
    import doctest
    doctest.testmod()
    
if __name__=='__main__':
    test()
