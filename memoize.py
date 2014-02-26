'''
Created on Jul 19, 2013

@author: Nathan Schneider (nschneid)
'''

def memoize(f):
    """
    Memoization decorator for a function taking one or more arguments.
    Source: http://code.activestate.com/recipes/578231-probably-the-fastest-memoization-decorator-in-the-/#c4 
    """
    class memodict(dict):
        def __getitem__(self, *key):
            return dict.__getitem__(self, key)

        def __missing__(self, key):
            ret = self[key] = f(*key)
            return ret

    return memodict().__getitem__
