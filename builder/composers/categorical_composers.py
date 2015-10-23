import sys

sys.path.append('.')
import logging
import os
from itertools import groupby
from operator import itemgetter
import numpy as np
import pandas as pd
from discoutils.thesaurus_loader import Vectors
from discoutils.tokens import DocumentFeature
from discoutils.misc import mkdirs_if_not_exists
from thesisgenerator.composers.vectorstore import (CopyObject, FrobeniusAdd, FrobeniusMult,
                                                   compose_and_write_vectors)

VERBS_HDF_DIR = '/lustre/scratch/inf/mmb28/FeatureExtractionToolkit/categorical/'
MIN_SVO_PER_VERB = 3  # todo does this filter exist in the original paper?


def train_verb_tensors(svos_file, noun_vectors_file, output_filename):
    """
    Trains Verb-bar matrices, as described in Milajevs et al (EMNLP-14, §3)
    :param svos_file: file containing a list of all SVOs in unlabelled data, one per line. May contain other document
     features too. Such a file is output by `find_all_NPs.py`, which is called from `observed_vectors.py`
    :param noun_vectors_file: a vector store containing noun vectors
    :param output_filename: name of output file- must identify the noun vectors and the unlabelled corpus
    """
    mkdirs_if_not_exists(os.path.dirname(output_filename))

    v = Vectors.from_tsv(noun_vectors_file)

    with open(svos_file) as infile:
        phrases = set()
        for line in infile:
            if DocumentFeature.from_string(line.strip()).type == 'SVO':
                phrases.add(tuple(line.strip().split('_')))
    phrases = [(subj, verb, obj) for subj, verb, obj in phrases if subj in v and obj in v]
    phrases = sorted(phrases, key=itemgetter(1))
    logging.info('Found %d SVOs in list', len(phrases))

    verb_tensors = dict()
    for verb, svos in groupby(phrases, itemgetter(1)):
        svos = list(svos)
        if len(svos) < MIN_SVO_PER_VERB:
            continue
        logging.info('Training matrix for %s from %d SVOs', verb, len(svos))
        vt = np.sum(np.outer(v.get_vector(subj).A, v.get_vector(obj).A) for subj, _, obj in svos)
        verb_tensors[verb] = vt

    logging.info('Trained %d verb matrices, saving...', len(verb_tensors))
    for verb, tensor in verb_tensors.items():
        df = pd.DataFrame(tensor)
        df.to_hdf(output_filename, verb.split('/')[0], complevel=9, complib='zlib')


def _nouns_only_filter(s, dfs):
    return dfs.type == '1-GRAM' and dfs.tokens[0].pos == 'N'


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s")

    FET = '/lustre/scratch/inf/mmb28/FeatureExtractionToolkit/'

    nouns_wins_wiki = os.path.join(FET, 'exp11-13b-ppmi/exp11-with-obs-phrases-SVD100.events.filtered.strings')
    nouns_w2v_gigaw_100 = os.path.join(FET, 'word2vec_vectors/word2vec-gigaw-100perc.unigr.strings.rep0')
    nouns_w2v_wiki_15 = os.path.join(FET, 'word2vec_vectors/word2vec-wiki-15perc.unigr.strings.rep0')
    nouns_w2v_wiki_100 = os.path.join(FET, 'word2vec_vectors/word2vec-wiki-100perc.unigr.strings.rep0')
    nouns_glove_wiki_100 = os.path.join(FET, 'glove/vectors.miro.h5')
    all_nouns = [nouns_wins_wiki, nouns_w2v_gigaw_100, nouns_w2v_wiki_15, nouns_w2v_wiki_100, nouns_glove_wiki_100]

    names_composed_files = ['wiki-wins-100', 'gigaw-w2v-100', 'wiki-w2v-15', 'wiki-w2v-100', 'wiki-glove-100']
    save_files = ['%s-vector-matrices.hdf' % x for x in names_composed_files]
    for noun_path, save_file, sname in zip(all_nouns,
                                           save_files,
                                           names_composed_files):
        trained_verb_matrices_file = os.path.join(VERBS_HDF_DIR, save_file)

        # the list of subject/objects of a given verb is determined from the unlabelled corpus,
        # and so are the noun vectors
        if 'wiki' in sname:
            svos_path = '/lustre/scratch/inf/mmb28/DiscoUtils/wiki_NPs_in_MR_R2_TechTC_am_maas.uniq.10.txt'
        elif 'giga' in sname:
            svos_path = '/lustre/scratch/inf/mmb28/DiscoUtils/gigaw_NPs_in_MR_R2_TechTC_am_maas.uniq.10.txt'
        else:
            raise ValueError('What unlabelled corpus is this???')
        train_verb_tensors(svos_path, noun_path, trained_verb_matrices_file)

        compose_and_write_vectors(noun_path,  # a vector store containing noun vectors
                                  sname,  # something to identify the source of unigram vectors
                                  [CopyObject, FrobeniusAdd, FrobeniusMult],
                                  # filename of output of training stage
                                  categorical_vector_matrix_file=trained_verb_matrices_file,
                                  output_dir=VERBS_HDF_DIR, dense_hd5=True)
