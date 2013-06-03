'''
Created on Sep 29, 2012

@author: Nathan Schneider (nschneid)
'''
from __future__ import absolute_import
import timeit

import pyximport; pyximport.install()
import discriminativeTagger

@timeit.Timer
def go():
    try:
        discriminativeTagger.main()
    except KeyboardInterrupt:
        raise

print go.timeit(number=1)
