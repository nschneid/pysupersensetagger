#!/usr/bin/env python2.7
'''
Enriches a file with just words, POSes, and supersense tags (and optionally IDs)
into the full 8- or 9-column format (which includes lemmas, parent indices, bare labels, etc.).
Lemmatization is performed using WordNet; the remaining columns are deterministic 
given the other information.

Arg: file

@author: Nathan Schneider (nschneid)
@since: 2015-03-15
'''
from __future__ import print_function
import sys
from dataFeaturizer import SupersenseDataSet

inData = SupersenseDataSet(sys.argv[1], labels=None, legacy0=False, keep_in_memory=False)
# SupersenseDataSet handles lemmatization

for sent in inData:
	# tags will be stored as gold; copy to predicted so they will be printed
	for i in range(len(sent)):
		sent[i] = sent[i]._replace(prediction=sent[i].gold)
	sent.updatedPredictions()
	print(sent)
	print()
