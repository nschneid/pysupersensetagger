PySupersenseTagger
==================

This is a Python port of Michael Heilman's [Java supersense tagger for English](http://www.ark.cs.cmu.edu/mheilman/questions/SupersenseTagger-05-17-11.tar.gz), 
which was a reimplementation of the system described in [Ciaramita and Altun, EMNLP 2006](http://www.aclweb.org/anthology/W06-1670).

PySupersenseTagger is currently IN DEVELOPMENT (pre-release).

Dependency: Python 2.7

Preparation: In the `data` directory, extract and preprocess the training data as follows:

```bash
gunzip SEM.BI.gz
./convert.pl SEM.BI > SEM.BI.data
```

Running: Currently, from the `src` directory, you can run `discriminativeTagger.py` with the following options:

```bash
--properties tagger.properties --labels ../data/SEM_07.BI.labels --train ../data/SEM.BI.data --iters 5 --debug --save testmodel.ser.gz
```

-- Nathan Schneider, 2012-09-10
