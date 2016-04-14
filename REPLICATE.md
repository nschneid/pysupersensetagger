# Replicating published results for the AMALGrAM system

Nathan Schneider  
2016-04-14

This document describes how to replicate the joint MWE+supersense tagging results in Schneider et al., NAACL-HLT 2015. (To replicate the MWE-only results in Schneider et al., *TACL* 2014, see instructions in README.md.)

1. Download [STREUSLE 2.0](http://www.cs.cmu.edu/~ark/LexSem/streusle-2.0.zip) instead of the most recent STREUSLE version.

2. Separate streusle.sst into  [train](http://www.cs.cmu.edu/~ark/LexSem/train.sentids)
and [test](http://www.cs.cmu.edu/~ark/LexSem/test.sentids) splits by sentence ID.

3. Run the following bash commands to remove the <code>\`</code> and <code>\`j</code> label refinements but keep the MWE part of the tag:

    ```bash
# keep `a (auxiliaries), remove `j and plain `
src/sst2tags.py $trainsst | sed -r $'s/-`j?\t/\t/g' > $traintags
src/sst2tags.py $testsst | sed -r $'s/-`j?\t/\t/g' > $testtags
cut -f5 $traintags | sort | uniq | sed '/^\s*$/d' > $tagset
```

4. Generate said.json, or if you do not have access to the SAID lexicon, you can create an empty file.

5. Download the pretrained model file [sst.model.pickle.gz](http://www.cs.cmu.edu/~ark/LexSem/sst.model.pickle.gz). Run `gunzip` to attempt to unzip it; if this produces an error, your web browser has probably unzipped it for you, so just remove the .gz extension from the filename. Then run predict_sst.sh on the test data.

6. You can retrain the model with the command below (with `$train` and `$test` data files), OR  and run `gunzip` if necessary.

        python2.7 src/main.py --cutoff 5 --iters 4 --YY tagsets/bio2gNV_dim --defaultY O --debug --train $train --test-predict $test --bio NO_SINGLETON_B --cluster-file mwelex/yelpac-c1000-m25.gz --clusters --lex mwelex/{semcor_mwes,wordnet_mwes,said,phrases_dot_net,wikimwe,enwikt}.json

    To save the learned model to a file, add `--save $modelfilename`.


## Expected results with said.json

For the condition in the paper which internally I call `Cutoff.5+FClust.1+FSet.sst+LearningCurve.1+Test.1`:

MWE scores (end of mweval.py output):

```
   P   |   R   |   F   |   EP  |   ER  |   EF  |  Acc  |   O   | non-O | ingap | B vs I | strength
 71.05%  56.24%  62.74%  63.98%  55.16%  59.22%  90.17%  96.91%  58.45%  98.38%  95.33%  100.00%
                                                   6466    6026     557     548     531     299
                                                   7171    6218     953     557     557     299
```

Supersense scores (end of ssteval.py output):

```
  Acc  |   P   |   R   |   F   || R: NSST | VSST |  `a  | PSST
 82.49%  69.47%  71.90%  70.67%    66.95%  74.17%  94.97%  nan%
   5915    1684    1684               798     735     151       0
   7171    2424    2342              1192     991     159       0
```

## Expected results without said.json

A user who trained and tested a model without SAID features (said.json) reports obtaining F1 scores of 61.37% for MWEs and 70.12% for supersenses.

## Acknowledgments

Thanks to Java Hosseini, Haibo Ding, and Youhyun Shin for requesting clarification on replicating the results, prompting this explanation.
