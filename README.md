[![Build Status](https://travis-ci.org/mbatchkarov/vector_builder.svg?branch=master)](https://travis-ci.org/mbatchkarov/vector_builder) [![Coverage Status](https://coveralls.io/repos/mbatchkarov/vector_builder/badge.svg?branch=master&service=github)](https://coveralls.io/github/mbatchkarov/vector_builder?branch=master)

A bunch of utilities for building distributional representations of words and phrases. This code mostly glues together other packages and converts their output to the [same format](https://github.com/mbatchkarov/DiscoUtils). The models included are those considered in my PhD work.

# Prerequisites

 - [DiscoUtils](https://github.com/mbatchkarov/DiscoUtils) as the common storage format
 - [dc_evaluation](https://github.com/mbatchkarov/dc_evaluation) to get lists of words and phrases of interest. Feel free to monkey-patch this.
 - [gensim](https://github.com/piskvorky/gensim/) for `word2vec` vectors
 - [glove](http://nlp.stanford.edu/projects/glove/) for `glove` vectors
 - configuration files and some text. Download [here](https://www.dropbox.com/s/w9b71zqy5o4tlcp/vector-builder-data.zip?dl=0).

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

```
python builder/get_word2vec_vectors.py --stages reformat vectors compose --corpus wiki 
```

The stages are:

 - reformat: take a directory of CoNLL files and output a directory with matching structure, where each sentence in a new line, and each token may include PoS tag (`--remove-pos` parameter). The input directory is assumed to be called e.g. `wiki-conll`, and the reformatted data goes to `wiki` or `wiki-nopos`.
 	- `wiki` is the name of the corpus, located in `data`
 - vectors: train word embeddings
 - compose: build phrase embeddings out of word embeddings.

The output of each stage is saved to disk, so you can interrupt the script at any time. Unigram vectors go to `outputs/word2vec/*unigr.strings.rep*` and composed phrase vectors to `outputs/word2vec/composed_word2vec*.events.filtered.strings`. You can now inspect your results:
```
from discoutils.thesaurus_loader import Vectors
v = Vectors.from_tsv('outputs/word2vec/wiki-100perc.unigr.strings.rep0')
v.init_sims(n_neighbors=5)
v.get_nearest_neighbours('spain/N')
```
outputs
```
Out[4]:
[('poland/N', 0.78751145271622558),
('austria/N', 0.81238981977584812),
('denmark/N', 0.83802344094440939),
('syria/N', 0.84054037191096609),
('france/N', 0.88849579829431435)]
```

> Note on speed: if `gensim` is installed correctly, training should go over >100k tokens a second. If training is slow, get a C compiler and re-install gensim.

### Other settings
The script also includes options that allow you to:

 - train on `x%` of all available files
 - train a model `x` times
 - average vectors across all repetitions

> Support for the `--remove-pos` flag is inconsistent. The `word2vec` script generally works well if words do not have a PoS tag, but other may fall over in odd ways. 

## glove

Download glove, then run `cp demo.sh demo2.sh`. In `demo2.sh`, comment out the bits where data is downloaded and a matlab evaluation script is run, and change `CORPUS=text8` to `CORPUS=$1`. Change other parameters you think are appropriate, like `VECTOR_SIZE` or `MAX_ITER`. Then run:
 
 ```
 python builder/get_glove_vectors.py --stages reformat vectors compose --corpus wiki --glove-dir /Volumes/LocalDataHD/m/mm/mmb28/projects/FeatureExtractionToolkit/glove
 # do not forget to change glove-dir
 # output of each stage is cached so it doesn't have to be repeated
 ```

 > GloVe may sometimes crash with a bus error. This is a bug in their C code and seems to occur with small training sets. Not sure what's going on there.

Unigram vectors will appear in `outputs/glove/vectors.wiki.h5` and composed vectors in 

Inspect your results:
```
from discoutils.thesaurus_loader import Vectors
v = Vectors.from_tsv('outputs/glove/vectors.wiki.h5')
v.init_sims(n_neighbors=5)
v.get_nearest_neighbours('spain/N')
```
outputs
```
Out[10]:
[('france/N', 1.0740608765386626),
 ('syria/N', 1.1164180411046751),
 ('persia/N', 1.1620328110298779),
 ('poland/N', 1.2090831826925723),
 ('wales/N', 1.2577004661074911)]
```

or 
```
v.get_nearest_neighbours('local/J_paper/N')
Out[14]:
[('local/J', 2.2867817977026146),
 ('business/N', 2.3535966696097272),
 ('private/J', 2.4526231745270208),
 ('society/N', 2.5353101049776932),
 ('create/V', 2.5608819354782839)]
 ```

 Note on input data:
 
 - One file, sentence per line. This is what `word2vec` reformatting produces. Reformatting puts the entire file on a single line. This is probably wrong.

## Counting vectors
 
First extract word co-occurrences with [FeatureExtractorToolkit](https://github.com/MLCL/FeatureExtractionToolkit) or a similar tool. The output needs to look like 

``` 
ENTRY	FEATURE1	FEATURE2
```

for instance


```
it/PRON	T:become/V	T:a/DET	T:full-time/J	nsubj-HEAD:tribunal/N
become/V	cop-HEAD:tribunal/N	T:it/PRON	T:./PUNCT
a/DET	T:1998/UNK	T:./PUNCT	det-HEAD:tribunal/N
```

The features can be anything, e.g. word co-occurrences, dependency parse features, etc. We generally prefix proximity/window features with a `T:` and dependency features with their type, e.g. `amod-HEAD:` or `amod-DEP:`. This shows the type of dependency relation and its direction.

Now count and build word with [Byblo](https://github.com/MLCL/Byblo):

```
python builder/build_phrasal_thesauri_offline.py --conf data/count_features/exp0.conf.txt  --byblo ../FeatureExtractionToolkit/Byblo-2.2.0 --stages unigrams ppmi svd compose  --use-ppmi
```

You will need:

 - a copy of Byble 2.2 and above
 - a Byblo configuration file

The file needs to be structured as follows:

```
--input
/Volumes/LocalDataHD/m/mm/mmb28/NetBeansProjects/vector_builder/data/count_features/example
--output
/Volumes/LocalDataHD/m/mm/mmb28/NetBeansProjects/vector_builder/outputs/count/
--threads
4
--filter-feature-freq
40
--filter-event-freq
15
```

The input file is what `FeatureExtractionToolkit` produced. Four additional directories will be created (one after each stage of the pipeline):

 - `outputs/count` : raw unweighted high-dimensional vectors. Find the `*events.filtered.strings` file.
 - `outputs/count-ppmi` : high-dimensional sparse vectors with PPMI reweighting. 
 - `outputs/count-ppmi-svd` : low-dimensional dense vectors
 - `outputs/count-ppmi-svd-composed` : low-dimensional composed phrasal vectors

## Socher et al vectors


## Random vectors
```
python builder/generate_random_vectors.py --output random_vect.h5 --dim 50
```