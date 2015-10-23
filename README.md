[![Build Status](https://travis-ci.org/mbatchkarov/vector_builder.svg?branch=master)](https://travis-ci.org/mbatchkarov/vector_builder)
[![Coverage Status](https://coveralls.io/repos/mbatchkarov/vector_builder/badge.svg?branch=master&service=github)](https://coveralls.io/github/mbatchkarov/vector_builder?branch=master)

A bunch of utilities for building distributional representations of words and phrases. This code mostly glues together other packages and converts their output to the [same format](https://github.com/mbatchkarov/DiscoUtils). The models included are those considered in my PhD work.

# Prerequisites

 - [DiscoUtils](https://github.com/mbatchkarov/DiscoUtils) as the common storage format
 - [gensim](https://github.com/piskvorky/gensim/) for `word2vec` vectors
 - [glove](http://nlp.stanford.edu/projects/glove/) for `glove` vectors
 - [dc_evaluation](https://github.com/mbatchkarov/dc_evaluation) to get lists of words and phrases of interest. Feel free to monkey-patch this.

# Building word vectors

If `dc_evaluation` is used to provide a list of words and phrases (henceforth called document features or just features) of interest, you must have run the compression stage with `--write-features` enabled. This should have produced a directory like:

```
features_in_labelled/
├── web_all_features.txt
├── web_np_modifiers.txt
├── web_socher.txt
└── web_verbs.txt
```

This contains the list of all features of interest.

## word2vec

## glove

## Counting vectors

## Random vectors
```
python builder/generate_random_vectors.py --output random_vect.h5 --dim 50
```

# Composition

## Pointwise models