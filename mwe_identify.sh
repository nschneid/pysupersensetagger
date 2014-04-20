#!/bin/bash

# Predict with an existing MWE model.
# Usage: ./mwe_identify.sh model input

set -eu
set -o pipefail

model=$1 # e.g.: mwe.model
input=$2 # word and POS tag on each line (tab-separated)

# predict MWEs with an existing model

python2.7 src/main.py --mwe --load $model --YY tagsets/bio2g --defaultY O --predict $input --bio NO_SINGLETON_B --clusters --cluster-file mwelex/yelpac-c1000-m25.gz --lex mwelex/{semcor_mwes,wordnet_mwes,said,phrases_dot_net,wikimwe,enwikt}.json /dev/null > $input.pred.tags

python2.7 src/tags2mwe.py $input.pred.tags > $input.pred.mwe
