'''
Created on Sep 29, 2012

@author: Nathan Schneider (nschneid)
'''
from __future__ import absolute_import, print_function
import timeit, sys

import pyximport; pyximport.install()
import discriminativeTagger

@timeit.Timer
def go():
    try:
        discriminativeTagger.main()
    except KeyboardInterrupt:
        raise

print(go.timeit(number=1), file=sys.stderr)
