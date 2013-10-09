#coding=utf-8
'''
Given one or more lexicons of multiword expressions 
and input sentences, constructs a lattice of possible 
MWEs in each sentence and searches for the shortest-path 
analysis.

Gappy expressions are allowed; the output label scheme is BbIiOo.

@author: Nathan Schneider (nschneid@cs.cmu.edu)
@since: 2013-07-22
'''
from __future__ import print_function, division

from mweFeatures import extractLexiconCandidates
from dataFeaturizer import SupersenseDataSet

from pyutil.corpus import mwe_lexicons

def segment(sent, max_gap_length):
    sentence_lemmas = [t.stem for t in sent]
    path, tags, tokinfo = mwe_lexicons._lexicons['all'].shortest_path_decoding(sentence_lemmas, max_gap_length=max_gap_length)
    return tokinfo
    #for toffset,tag,expr_tokens,is_gappy_expr,entry in tokinfo:
    #    print(tag)
    

def main():
    import argparse
    
    opts = argparse.ArgumentParser(description='Learn or predict from a discriminative tagging model')
    
    def flag(name, description, ftype=str, **kwargs):
        opts.add_argument(('--' if len(name)>1 else '-')+name, type=ftype, help=description, **kwargs)
    def inflag(name, description, ftype=argparse.FileType('r'), **kwargs):
        flag(name, description, ftype=ftype, **kwargs)
    def outflag(name, description, ftype=argparse.FileType('w'), **kwargs):
        flag(name, description, ftype=ftype, **kwargs)
    def boolflag(name, description, default=False, **kwargs):
        opts.add_argument(('--' if len(name)>1 else '-')+name, action='store_false' if default else 'store_true', help=description, **kwargs)
    
    flag("data", "Path to training data feature file")  #inflag
    boolflag("legacy0", "BIO scheme uses '0' instead of 'O'")
    inflag("lex", "Lexicons to load for lookup features", nargs='*')
    flag("max-gap-length", "Maximum number of tokens within a gap", ftype=int, default=2)
    
    args = opts.parse_args()
    
    mwe_lexicons.load_combined_lexicon('all', args.lex)
    
    for sent in SupersenseDataSet(args.data, list('OoBbIiĪīĨĩ'.decode('utf-8')), legacy0=False):
        for tok,tokinfo in zip(sent,segment(sent, max_gap_length=args.max_gap_length)):
            gold = tok.gold.replace('ī'.decode('utf-8'),'i') \
                           .replace('ĩ'.decode('utf-8'),'i') \
                           .replace('Ī'.decode('utf-8'),'I') \
                           .replace('Ĩ'.decode('utf-8'),'I')
            if tokinfo[1]=='O':
                result = '*MISS*' if gold!='O' else ''
            elif tokinfo[1]!=gold:
                result = '*WRONG*' if gold!='O' else '*EXTRA*'
            else:
                result = '*CORRECT*'
            print(tok.token, tok.stem, gold, tokinfo[1], result,
                  ('' if not tokinfo[-1] else tokinfo[-1]["datasource"]), sep='\t')
        print()
        #assert False

if __name__ == '__main__':
    main()
