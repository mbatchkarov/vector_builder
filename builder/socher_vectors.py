from collections import defaultdict
import string
import sys
from nltk import WordNetLemmatizer

sys.path.append('.')
import argparse
import logging
from operator import itemgetter
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import loadmat
from discoutils.tokens import DocumentFeature
from discoutils.io_utils import write_vectors_to_hdf
from discoutils.cmd_utils import run_and_log_output
from discoutils.misc import temp_chdir, mkdirs_if_not_exists
from discoutils.thesaurus_loader import DenseVectors
from thesisgenerator.composers.vectorstore import (MultiplicativeComposer, AdditiveComposer,
                                                   RightmostWordComposer, LeftmostWordComposer,
                                                   VerbComposer, compose_and_write_vectors)
from thesisgenerator.utils.misc import force_symlink

# SET UP A FEW REQUIRED PATHS
# where are the composed n-gram vectors, must contain parsed.txt, phrases.txt and outVectors.txt
# before running this function, put all phrases to be composed in parsed.txt, wrapping them
# up to make them look like fragment of a syntactic parser. Do NOT let the Stanford parser that ships with
# that code run.
prefix = '/lustre/scratch/inf/mmb28/FeatureExtractionToolkit'
socher_base_dir = os.path.join(prefix, 'socher_vectors')  # copy downloaded content here

# the two paths below needs to point to the same thing
phrases_to_compose = os.path.join(prefix, '..', 'thesisgenerator',
                                  'features_in_labelled', 'socher.txt')
socher_input_file = os.path.join(socher_base_dir, 'parsed.txt')
plaintext_socher_input_file = os.path.join(prefix, '..', 'thesisgenerator',
                                           'features_in_labelled', 'all_features.txt')

socher_output_phrases_file = os.path.join(socher_base_dir, 'phrases.txt')
socher_output_vectors_file = os.path.join(socher_base_dir, 'outVectors.txt')
socher_unigram_embedding_matlab = os.path.join(socher_base_dir, 'vars.normalized.100.mat')

# output of reformat stage
turian_unigram_vectors_file = os.path.join(socher_base_dir, 'turian_unigrams.h5')
output_dir = os.path.join(socher_base_dir, 'composed')
mkdirs_if_not_exists(output_dir)
socher_composed_vectors_file = os.path.join(output_dir, 'AN_NN_turian_Socher.events.filtered.strings')


def run_socher_code():
    # symlink the file Socher's code expects to where the list of phrases I'm interested is
    force_symlink(phrases_to_compose, socher_input_file)
    with temp_chdir(socher_base_dir):
        run_and_log_output('./phrase2Vector.sh')  # this takes a while
        # output files are phrases.txt and outVectors.txt


def reformat_socher_vectors():
    """
    Formats the files output by Socher (2011)'s matlab code into byblo-compatible files.

    Before running this a list of all phrases needs to be extracted from the labelled data, and these need to
    be composed with Socher's matlab code. See note "Socher vectors" in Evernote.

    """
    logging.info('Reformatting events file %s ---> %s',
                 socher_output_vectors_file, socher_composed_vectors_file)

    # socher's code removes all PoS tags, so we can't translate his output
    # back to a DocumentFeature. Let's read the input to his code instead and
    # get the corresponding output vectors
    # get a list of all phrases that we attempted to compose
    with open(plaintext_socher_input_file) as infile:
        composed_phrases = [DocumentFeature.from_string(line.strip()) for line in infile]

    # get a list of all phrases where composition worked (no unknown words)
    with open(socher_output_phrases_file) as infile:
        success = [i for i, line in enumerate(infile) if '*UNKNOWN*' not in line]
        # pick out just the phrases that composes successfully
    composed_phrases = itemgetter(*success)(composed_phrases)

    # load all vectors, remove these containing unknown words
    mat = np.loadtxt(socher_output_vectors_file, delimiter=',')
    mat = mat[success, :]
    assert len(composed_phrases) == mat.shape[0]  # same number of rows

    # do the actual writing
    write_vectors_to_hdf(sp.coo_matrix(mat),
                         composed_phrases,
                         ['RAE-feat%d' % i for i in range(100)],  # Socher provides 100-dimensional vectors
                         socher_composed_vectors_file)


def write_clean_turian_unigrams():
    """
    Extracts unigram embeddings from Socher's binary distribution. These can be used by other composers.

    There are only 50k embeddings (presumably for the most frequent tokens in the corpus). The words have not
    been processed- there are punctuation-only tokens, uppercased words and non-lemmatized words. There isn't
    any PoS tag filtering either- words like "to", "while" and "there".

    I remove punctuation, then lowercase and lemmatize each entry. Multiple entries may map to the
    same canonical form. I select the shortest original entry (ties are broken by giving preference to
    words that are already lowercased). This could have been done better.
    Only vectors for the selected entries are kept. There's 33k canonical
    forms left, many of which are not nouns/adjs/verbs.

    We don't have a PoS tag for the canonical forms. I get around the problem by creating 3 copies of each
    canonical form and expand "cat" to cat/N, cat/J and cat/V, which all share the same vector.
    """
    logging.info('Writing Turian unigrams to %s', turian_unigram_vectors_file)
    mat = loadmat(socher_unigram_embedding_matlab)
    words = [w[0] for w in mat['words'].ravel()]
    df = pd.DataFrame(mat['We'].T, index=words)

    lmtzr = WordNetLemmatizer()
    clean_to_dirty = defaultdict(list)  # canonical -> [non-canonical]
    dirty_to_clean = dict()  # non-canonical -> canonical
    to_keep = set()  # which non-canonical forms forms we will keep
    #  todo this can be done based on frequency or something

    for w in words:
        if set(w).intersection(set(string.punctuation).union(set('0123456789'))):
            # not a real word- contains digits or punctuation
            continue

        lemma = lmtzr.lemmatize(w.lower())
        clean_to_dirty[lemma].append(w)
        dirty_to_clean[w] = lemma

    # decide which of possibly many non-canonical forms with the same lemma to keep
    # prefer shorter and lowercased non-canonical forms
    for lemma, dirty_list in clean_to_dirty.items():
        if len(dirty_list) > 1:
            best_lemma = min(dirty_list, key=lambda w: (len(w), not w.islower()))
        else:
            best_lemma = dirty_list[0]
        to_keep.add(best_lemma)

    # remove non-canonical forms we don't want
    idx_to_drop = [i for i, w in enumerate(df.index) if w not in to_keep]
    ddf = df.drop(df.index[idx_to_drop])
    # canonicalize whatever is left
    ddf.index = [lmtzr.lemmatize(w.lower()) for w in ddf.index]

    # we don't know what the PoS tags of the canonical forms are, so make them all of the same tag
    # e.g. expand "cat" to cat/N, cat/J and cat/V, which all share the same vector
    new_index = ['%s/%s'%(w, pos) for pos in 'NJV' for w in ddf.index]
    new_data = np.vstack([ddf.values] * 3)
    ddf = pd.DataFrame(new_data, index= new_index)
    dv = DenseVectors(ddf, allow_lexical_overlap=True)
    dv.to_tsv(turian_unigram_vectors_file)
    logging.info('Done')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', choices=('fancy-compose', 'format', 'simple-compose'), nargs='+', required=True,
                        help="""Stages are as follows:
                         - fancy-compose: runs Socher's code (Turian unigrams and Socher composition)
                         - format: converts output of previous stage to Byblo-compatible files
                         - simple-compose: does Add/Mult... composition on Turian unigrams, as converted in
                         previous stage
                        """)
    args = parser.parse_args()

    if 'fancy-compose' in args.stages:
        run_socher_code()
    if 'format' in args.stages:
        # write just the unigram vectors for other composers to use
        write_clean_turian_unigrams()
        reformat_socher_vectors()
    if 'simple-compose' in args.stages:
        composers = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer,
                     RightmostWordComposer, VerbComposer]
        compose_and_write_vectors(turian_unigram_vectors_file,
                                  'turian',
                                  composers,
                                  output_dir=output_dir, gzipped=False, dense_hd5=True)
