'''
Created on Sep 29, 2012

@author: Nathan Schneider (nschneid)
'''
import pyximport; pyximport.install()
import discriminativeTagger


try:
    discriminativeTagger.main()
except KeyboardInterrupt:
    raise
