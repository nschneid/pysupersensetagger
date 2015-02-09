#!/bin/bash

# Train and test the MWE identification system
# This requires the CMWE corpus; see README.md to download 
# and follow the instructions to restore the full text from the LDC data.
# (To predict with an existing model, see mwe_identify.sh.)

set -eu
set -o pipefail

ark=ark-tweet-nlp-0.3.2

# - prepare train and test data (automatic POS tags)

        # look up entries by sentence id to populate the train and test splits
        SCRIPT="from __future__ import print_function
import fileinput, sys
corpusFP, splitFP = sys.argv[1:]
entries = {}
for ln in fileinput.input(sys.argv[1]):
  sentid, rest = ln.split('\t',1)
  assert sentid not in entries
  entries[sentid] = rest.strip()
for ln in fileinput.input(sys.argv[2]):
  sentid = ln.strip()
  print(sentid, entries[sentid], sep='\t')
" 
	python2.7 -c "$SCRIPT" cmwe/corpus.mwe train.sentids > cmwe/train.mwe
	python2.7 -c "$SCRIPT" cmwe/corpus.mwe test.sentids > cmwe/test.mwe

	# convert to tags

	python2.7 src/mwe2tags.py cmwe/train.mwe > cmwe/train.tags
	python2.7 src/mwe2tags.py cmwe/test.mwe > cmwe/test.tags

	#train the POS tagger:
	#java -XX:ParallelGCThreads=2 -Xmx8g -cp $ark/ark-tweet-nlp-0.3.2.jar cmu.arktweetnlp.Train $in ewtb_pos.model
	
	# prepare POS tagger input
	cut -f2,4,5,9 cmwe/train.tags > cmwe/train.wdposid
	cut -f2,4,5,9 cmwe/test.tags > cmwe/test.wdposid

	# run POS tagger
	$ark/runTagger.sh --input-format conll --output-format conll --model ewtb_pos.model cmwe/train.wdposid | cut -f1-2 > cmwe/train.syspos.wdposid
	paste cmwe/train.syspos.wdposid <(cut -f3-4 cmwe/train.wdposid) > cmwe/train.syspos.wdposid
	$ark/runTagger.sh --input-format conll --output-format conll --model ewtb_pos.model cmwe/test.wdposid | cut -f1-2 > cmwe/test.syspos.wdposid
	paste cmwe/test.syspos.wdposid <(cut -f3-4 cmwe/test.wdposid) > cmwe/test.syspos.wdposid

	# incorporate system POS tags
	paste <(cut -f1-3 cmwe/train.tags) <(cut -f2 cmwe/train.syspos.wdposid) <(cut -f5- cmwe/train.tags) > cmwe/train.syspos.tags
	paste <(cut -f1-3 cmwe/test.tags) <(cut -f2 cmwe/test.syspos.wdposid) <(cut -f5- cmwe/test.tags) > cmwe/test.syspos.tags



# - learning on training set, predict on test set

python2.7 src/main.py --mwe --YY tagsets/bio2g --defaultY O --train cmwe/train.syspos.tags --test-predict cmwe/test.syspos.tags --iters 3 --debug --save mwe.model --bio NO_SINGLETON_B --clusters --cluster-file mwelex/yelpac-c1000-m25.gz --lex mwelex/{semcor_mwes,wordnet_mwes,said,phrases_dot_net,wikimwe,enwikt}.json /dev/null --includeLossTerm --costAug 100  > testpredictions.syspos.tags
# the /dev/null was supposed to be ignored, but it does have a small effect

# - convert predictions to .mwe

python2.7 src/tags2mwe.py testpredictions.syspos.tags > testpredictions.syspos.mwe

# - evaluate

python2.7 src/tags2mwe.py cmwe/test.tags > cmwe/test.withtags.mwe

python2.7 src/mweval.py --default-strength strong cmwe/test.withtags.mwe testpredictions.syspos.mwe > testpredictions.eval
