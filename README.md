AMALGr 1.0
==========

__AMALGr__ (**A** <b>M</b>achine <b>A</b>nalyzer of <b>L</b>exical <b>Gr</b>oupings) identifies a broad range of English multiword expressions (MWEs) in context. For example, given the sentence

> I do n't hold a grudge against him over his views , which I think have a lot to do with his upbringing in El Paso .

the analysis

```
I do n't hold_a_grudge against him over his views , which I think have_ a_lot _to_do with his upbringing in El_Paso .
```

will be predicted, grouping "hold a grudge", "have to do", "a lot", and "El Paso" as MWEs. The model and algorithms implemented in the tool are described in Schneider et al. (*TACL* 2014), and resources required to use it are available at http://www.ark.cs.cmu.edu/LexSem/.

More generally, the codebase supports supervised discriminative learning and structured prediction of statistical sequence models over discrete data, i.e., taggers. It implements the structured perceptron (Collins, EMNLP 2002) for learning and the Viterbi algorithm for decoding. Cython is used to make decoding reasonably fast, even with millions of features. The software is released under the GPLv3 license (see LICENSE file).


Obtaining the code
------------------

To run this software you will need to [download the source code](http://www.ark.cs.cmu.edu/download.php?url=https://github.com/nschneid/pysupersensetagger/archive/v1.0.zip) (no binaries are available).
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
  - NLTK 2.0.4 with the WordNet resource installed

The MWE Identifier requires the input to be sentence and word tokenized and part-of-speech tagged (with the Penn Treebank POS tagset). To obtain automatic POS tags for tokenized text, we recommend the TurboTagger module within [TurboParser](http://www.ark.cs.cmu.edu/TurboParser/) or the [TweetNLP Tagger](http://www.ark.cs.cmu.edu/TweetNLP/).

### Data

#### Lexicons and Word Clusters

Features in the MWE Identification model make use of several MWE lists extracted from existing English lexicons, as well as word clusters from a corpus of Yelp reviews. These are available as a separate download at http://www.ark.cs.cmu.edu/LexSem/.

#### Corpus

The annotated data that was used to train and evaluate the MWE Identifier depends on the English Web Treebank, which is distributed by LDC. http://www.ark.cs.cmu.edu/LexSem/ provides stand-off annotations and a script to merge them with the full text. This corpus is *not* required to identify MWEs in other data with a pretrained model.


Installation Instructions
-------------------------

The necessary lexical semantic resources and software are linked from http://www.ark.cs.cmu.edu/LexSem/.

0. Make sure your system has the software described in the previous section.
1. Download and unzip the AMALGr software release.
2. Download and gunzip the MWE identification model in the AMALGr main directory.
3. Download and unzip the English Multiword Expression Lexicons in the AMALGr main directory.
4. If you have access to the SAID resource from LDC, follow the instructions in the lexicons package to build the said.json lexicon. Otherwise, create an empty file in its place:

        $ touch mwelex/said.json

    Note that the system will be at a disadvantage if it was trained to leverage SAID but cannot access it.
5. Check that the AMALGr main directory contains the directories and src/ and mwelex/ and the file mwe.model.pickle.
6. *Required for learning with CMWE, not required for prediction*: Download and unzip the CMWE Corpus in the AMALGr main directory. If you have access to the English Web Treebank from LDC, follow the instructions in the CMWE package to complete the corpus.
7. *Required to emulate the* TACL *experiments*: Download train.sentids and test.sentids into the AMALGr main directory.


Prediction with a Pretrained Model
----------------------------------

### Input Format

The input to AMALGr (for MWE identification) must contain one line per token, with a blank line in between sentences. Each line must have 2, 3, or 4 tab-separated fields:

1. token
2. POS tag
3. OPTIONAL: gold MWE tag against which to measure accuracy
4. OPTIONAL: an identifier for the sentence (will be retained in the output)

A few example POS-tagged sentences are given in the file: example


### Execution

On the command line, call the `mwe_identify.sh` script with the name of the model and the path to the POS-tagged input file:

    ./mwe_identify.sh mwe.model example

This will create two output files: example.pred.tags and example.pred.mwe.

Note: on first execution, you may see compiler warnings.

### Output Format

#### .tags

The format output directly by the tool consists of one line per token with 9 tab-separated fields on each line:

1. offset (1-based)
2. token
3. lemma
4. POS tag
5. MWE tag: one of O o B b Ī Ĩ ī ĩ (see Schneider et al., *TACL* 2014 for an explanation)
6. MWE parent offset, or 0 if not continuing an MWE
7. MWE attachment strength: _ (strong) or ~ (weak)
8. (empty placeholder)
9. sentence ID, if present in input

#### .mwe

For many purposes, it is more practical to have a more unified encoding of each sentence and its analysis. AMALGr includes a scripts to convert between the .tags format and a .mwe format, which represents one sentence per line. (See src/mwe2tags.py and src/tags2mwe.py.)

The .mwe format contains 3 tab-separated columns:

1. sentence ID, if present in input
2. human-readable MWE analysis of the sentence
3. a JSON object containing:
   - `words`: the tokens and their POS tags
   - `lemmas`
   - `tags`: predicted MWE tags
   - `_`: strong MWE groupings of token offsets
   - `~`: weak MWE groupings of token offsets (including all tokens of strong MWEs that are constituents of weak MWEs)

Note that token offsets are 1-based (as in the .tags format).

To see just the human-readable output, use:

    $ cut -f2 example.pred.mwe
    Sounds haunting , and a little bit Harry_Potter , but I want to check_ it _out .
    YEAR 3 : Hermione held her own when Draco_Malfoy and his minions gave_ her _grief .
    The silver_screen adventures of Harry , Ron and Hermione have been a magic pot of gold for Hollywood studio Warner_Bros , with the seven films released so_far grossing $ 6.4 billion in ticket_sales and billions more from DVDs and merchandise .

Learning and Evaluation
-----------------------

The script train_test.sh executes the full pipeline of corpus preprocessing, POS tagging, training, test set prediction, and evaluation that constitutes the best experimental setting with automatic POS tags in (Schneider et al., *TACL* 2014). To replicate it, you will need the CMWE corpus as well as the ARK TweetNLP POS Tagger and ewtb_pos.model. You will have to edit the `ark` variable in train_test.sh to point to the POS tagger directory.


Contributing
------------

Development of this software is tracked [on GitHub](https://github.com/nschneid/pysupersensetagger). To contribute, open an issue or submit a pull request. See DEVELOPER for a note about the pyutil subdirectory.


History
-------

  - 1.0: 2014-04-20

This software was developed by [Nathan Schneider](http://nathan.cl).

The codebase originated as a Python port of Michael Heilman's [Java supersense tagger for English](http://www.ark.cs.cmu.edu/mheilman/questions/SupesenseTagger-05-17-11.tar.gz), 
which was a reimplementation of the system described in Ciaramita and Altun (EMNLP 2006).

