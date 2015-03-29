#!/bin/bash
set -eu

# Generate supersense lexicon from WordNet

mkdir lex

python2.7 src/sstFeatures.py
