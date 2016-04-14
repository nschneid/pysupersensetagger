AMALGrAM 2.0
============

__AMALGrAM__ (**A** <b>M</b>achine <b>A</b>nalyzer of <b>L</b>exical <b>Gr</b>oupings <b>A</b>nd <b>M</b>eanings) analyzes English sentences for multiword expressions (MWEs) and noun and verb supersenses. For example, given the sentence

> I do n't think he 's afraid to take a strong stand on gun control , what with his upbringing in El Paso .

the analysis

```
I do|`a n't think|cognition he 's|stative afraid to take_a_ strong _stand|cognition on gun_control|ARTIFACT , what_with his upbringing|ATTRIBUTE in El_Paso|LOCATION .
```

will be predicted, grouping "take a stand", "gun control", "what with", and "El Paso" as MWEs and labeling several lexical expressions with supersenses (UPPERCASE for nouns, lowercase for verbs). The model and algorithms implemented in the tool are described in Schneider et al. (*TACL* 2014, NAACL-HLT 2015), and resources required to use it are available at http://www.cs.cmu.edu/~ark/LexSem/.

More generally, the codebase supports supervised discriminative learning and structured prediction of statistical sequence models over discrete data, i.e., taggers. It implements the structured perceptron (Collins, EMNLP 2002) for learning and the Viterbi algorithm for decoding. Cython is used to make decoding reasonably fast, even with millions of features. The software is released under the GPLv3 license (see LICENSE file).


Obtaining the code
------------------

To run this software you will need to [download the source code](https://github.com/nschneid/pysupersensetagger/archive/master.zip) (no binaries are available).
The code will automatically be compiled from source when it is first run.

The latest development version can also be obtained via from GitHub (see [Contributing](#contributing)).


Dependencies
------------

### Platform

This software has been tested on recent Unix and Mac OS X platforms.
It has *not* been tested on Windows.

### Software

  - Python 2.7
  - Cython (tested on 0.19.1)
  - NLTK 3.0.2+ with the WordNet resource installed

The input must be sentence and word tokenized and part-of-speech tagged (with the Penn Treebank POS tagset). To obtain automatic POS tags for tokenized text, we recommend the TurboTagger module within [TurboParser](http://www.cs.cmu.edu/~ark/TurboParser/) or the [TweetNLP Tagger](http://www.cs.cmu.edu/~ark/TweetNLP/).

### Data

#### Lexicons and Word Clusters

Features in AMALGrAM's tagging model make use of several MWE lists extracted from existing English lexicons, as well as word clusters from a corpus of Yelp reviews. These are available as a separate download at http://www.cs.cmu.edu/~ark/LexSem/.

#### Corpus

The sentences in the annotated dataset that was used to train and evaluate AMALGrAM come from the English Web Treebank (EWTB), which is distributed by LDC. With permission from LDC and Google, the STREUSLE download at http://www.cs.cmu.edu/~ark/LexSem/ includes the source sentences and gold POS tags, but not the parse trees, from EWTB. The parse trees were not used in the lexical semantic annotation or in training the AMALGrAM tagger.


Installation Instructions
-------------------------

The necessary lexical semantic resources and software are linked from http://www.cs.cmu.edu/~ark/LexSem/.

0. Make sure your system has the software described in the previous section.
1. Download and unzip the AMALGrAM software release.
2. Download and gunzip the tagger model in the AMALGrAM main directory. (If gunzip fails, your web browser may have unzipped it for you; simply remove the .gz extension from the filename.)
3. Download and unzip the English Multiword Expression Lexicons in the AMALGrAM main directory.
4. If you have access to the SAID resource from LDC, follow the instructions in the lexicons package to build the said.json lexicon. Otherwise, create an empty file in its place:

        $ touch mwelex/said.json

    Note that the system will be at a disadvantage if it was trained to leverage SAID but cannot access it.
5. Check that the AMALGrAM main directory contains the directories and src/ and mwelex/ and the file sst.model.pickle.
6. From the main directory, run the following script to generate a supersense lexicon from WordNet (via NLTK):

        $ ./install.sh

7. *Required for learning with STREUSLE, not required for prediction*: Download and unzip the STREUSLE corpus in the AMALGrAM main directory.


Prediction with a Pretrained Model
----------------------------------

### Input Format

The input to AMALGrAM (for MWE identification) must contain one line per token, with a blank line in between sentences. Each line must have 2, 3, or 4 tab-separated fields:

1. token
2. POS tag
3. OPTIONAL: gold MWE+supersense tag against which to measure accuracy
4. OPTIONAL: an identifier for the sentence (will be retained in the output)

A few example POS-tagged sentences are given in the file: example


### Execution

On the command line, call the sst.sh script with the path to the POS-tagged input file:

    $ ./sst.sh example

This will create two output files: example.pred.tags and example.pred.sst.

Note: On first execution, you may see compiler warnings.

### Output Format

#### .tags

The format output directly by the tool consists of one line per token with 9 tab-separated fields on each line:

1. offset (1-based)
2. token
3. lemma
4. POS tag
5. MWE+supersense tag: one of O o B b Ī Ĩ ī ĩ (see Schneider et al., *TACL* 2014 for an explanation), possibly hyphenated with a supersense label. E.g.: O-COGNITION
6. MWE parent offset, or 0 if not continuing an MWE
7. MWE attachment strength: _ (strong) or ~ (weak)
8. supersense label: lowercase for verb supersenses and UPPERCASE for noun supersenses
9. sentence ID, if present in input

#### .sst

For many purposes, it is more practical to have a more unified encoding of each sentence and its analysis. AMALGrAM includes scripts to convert between the .tags format and a .sst format, which represents one sentence per line. (See src/sst2tags.py and src/tags2sst.py.)

The .sst format contains 3 tab-separated columns:

1. sentence ID, if present in input
2. human-readable MWE(+supersense) analysis of the sentence (supersense labels will only be included here if tags2sst.py is run with the -l option)
3. a JSON object containing:
   - `words`: the tokens and their POS tags
   - `lemmas`
   - `tags`: predicted MWE+supersense tags
   - `labels`: mapping from token offset to supersense label (for strong MWEs, it is only the first token in the MWE)
   - `_`: strong MWE groupings of token offsets
   - `~`: weak MWE groupings of token offsets (including all tokens of strong MWEs that are constituents of weak MWEs)

Note that token offsets are 1-based (as in the .tags format).

To see just the human-readable output, use:

    $ cut -f2 example.pred.mwe
    Sounds|`a haunting|motion , and a little bit|TIME Harry_Potter|GROUP , but I want|cognition to check_ it _out|social .
    YEAR|TIME 3 : Hermione held|social her own when Draco_Malfoy|PERSON and his minions|PERSON gave_ her _grief|possession .
    The silver_screen|ARTIFACT adventures|ACT of Harry|LOCATION~,~Ron|LOCATION and Hermione|PERSON have|`a been|stative a magic pot|QUANTITY of gold|POSSESSION for Hollywood_studio|GROUP Warner_Bros|LOCATION , with the seven films|COMMUNICATION released|change so_far grossing|possession $ 6.4 billion in ticket|POSSESSION sales|POSSESSION and billions|ARTIFACT more from DVDs|ARTIFACT and merchandise|ARTIFACT .

Learning and Evaluation
-----------------------

The evaluation scripts are src/mweval.py (for segmentation only) and src/ssteval.py (for supersense labels only).

MWE+supersense model: The default model (sst.model) was trained with the hyperparameter settings: `--cutoff 5 --iters 4` (these were tuned by cross-validation on the training data). See (Schneider et al., NAACL-HLT 2015) for details, and REPLICATE.md for detailed instructions on replicating the main results from that paper.

MWE-only model: The script train_test_mwe.sh executes the full pipeline of corpus preprocessing, POS tagging, training, test set prediction, and evaluation that constitutes the best experimental setting with automatic POS tags in (Schneider et al., *TACL* 2014). To replicate it, you will need the CMWE 1.0 corpus as well as the ARK TweetNLP POS Tagger and ewtb_pos.model. You will have to edit the `ark` variable in train_test_mwe.sh to point to the POS tagger directory.


Contributing
------------

Development of this software is tracked [on GitHub](https://github.com/nschneid/pysupersensetagger). To contribute, open an issue or submit a pull request. See DEVELOPER for a note about the pyutil subdirectory.


History
-------

  - 2.0: 2015-03-29. Added noun and verb supersense tagging (AMALGrAM).
  - 1.0: 2014-04-20. Multiword expression tagger (AMALGr).

This software was developed by [Nathan Schneider](http://nathan.cl).

The codebase originated as a Python port of Michael Heilman's [Java supersense tagger for English](https://github.com/kutschkem/SmithHeilmann_fork/tree/master/MIRATagger),
which was a reimplementation of the system described in Ciaramita and Altun (EMNLP 2006).
