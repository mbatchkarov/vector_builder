import os
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from os.path import join, basename
from glob import glob
import argparse
import logging
from discoutils.misc import mkdirs_if_not_exists
from discoutils.cmd_utils import (set_stage_in_byblo_conf_file, run_byblo, parse_byblo_conf_file,
                                  unindex_all_byblo_vectors, get_byblo_out_prefix)
from discoutils.reweighting import ppmi_sparse_matrix
from discoutils.reduce_dimensionality import do_svd
from discoutils.misc import temp_chdir
from discoutils.thesaurus_loader import Vectors
from thesisgenerator.composers.vectorstore import (AdditiveComposer, MultiplicativeComposer,
                                                   LeftmostWordComposer, RightmostWordComposer,
                                                   BaroniComposer, GuevaraComposer, GrefenstetteMultistepComposer,
                                                   compose_and_write_vectors, VerbComposer)
from thesisgenerator.composers.baroni_group import (train_baroni_guevara_composers,
                                                    train_grefenstette_multistep_composer)

"""
Composed wiki/gigaw dependency/window vectors and writes them to FeatureExtractionToolkit/exp10-13-composed-ngrams
"""
prefix = '/lustre/scratch/inf/mmb28/FeatureExtractionToolkit/'
byblo_base_dir = join(prefix, 'Byblo-2.2.0')


def calculate_unigram_vectors(thesaurus_dir):
    # find conf file in directory
    byblo_conf_file = _find_conf_file(thesaurus_dir)

    # find out where the conf file said output should go
    opts, _ = parse_byblo_conf_file(byblo_conf_file)
    byblo_output_prefix = join(opts.output, basename(opts.input))

    # get byblo to calculate vectors for all entries
    set_stage_in_byblo_conf_file(byblo_conf_file, 1)
    run_byblo(byblo_conf_file)
    set_stage_in_byblo_conf_file(byblo_conf_file, 0)
    # get vectors as strings
    unindex_all_byblo_vectors(byblo_output_prefix)


def _find_conf_file(thesaurus_dir):
    try:
        return glob(join(thesaurus_dir, '*conf*'))[0]
    except IndexError:
        return None  # no conf file in that dir


def _find_events_file(thesaurus_dir):
    conf_file = _find_conf_file(thesaurus_dir)
    if conf_file:
        return get_byblo_out_prefix(conf_file) + '.events.filtered.strings'
    else:
        return [x for x in glob(join(thesaurus_dir, '*events.filtered.strings')) if 'svd' not in x.lower()][0]


def _find_output_prefix(thesaurus_dir):
    # todo this will not work if we're throwing multiple events/features/entries files (eg SVD reduced and non-reduced)
    # into the same directory
    return os.path.commonprefix(
        [x for x in glob(join(thesaurus_dir, '*filtered*')) if 'svd' not in x.lower()])[:-1]


def _do_ppmi(vectors_path, output_dir):
    v = Vectors.from_tsv(vectors_path)
    ppmi_sparse_matrix(v.matrix)
    v.to_tsv(join(output_dir, basename(vectors_path)), gzipped=True)


def build_full_composed_thesauri_with_baroni_and_svd(corpus, features, stages):
    # SET UP A FEW REQUIRED PATHS
    global prefix, byblo_base_dir
    # INPUT 1:  DIRECTORY. Must contain a single conf file
    unigram_thesaurus_dir = join(prefix, 'exp%d-%db' % (corpus, features))
    unigram_thesaurus_dir_ppmi = os.path.abspath(join(prefix, 'exp%d-%db-ppmi' % (corpus, features)))
    mkdirs_if_not_exists(unigram_thesaurus_dir_ppmi)

    # INPUT 2: A FILE, TSV, underscore-separated observed vectors for ANs and NNs
    SVD_DIMS = 100
    dataset_name = 'gigaw' if corpus == 10 else 'wiki'  # short name of input corpus
    features_name = 'wins' if features == 13 else 'deps'  # short name of input corpus
    observed_phrasal_vectors = join(prefix, 'observed_vectors',
                                    '%s_NPs_%s_observed' % (dataset_name, features_name))
    ngram_vectors_dir = join(prefix,
                             'exp%d-%d-composed-ngrams-ppmi-svd' % (corpus, features))  # output 1
    if features_name == 'wins':
        composer_algos = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer, VerbComposer,
                          RightmostWordComposer, BaroniComposer, GuevaraComposer]
    else:
        # can't train Baroni/Guevara on deps because I don't have observed vectors for phrases
        composer_algos = [AdditiveComposer, MultiplicativeComposer, VerbComposer,
                          LeftmostWordComposer, RightmostWordComposer]

    mkdirs_if_not_exists(ngram_vectors_dir)

    # EXTRACT UNIGRAM VECTORS WITH BYBLO
    if 'unigrams' in stages:
        with temp_chdir(byblo_base_dir):
            calculate_unigram_vectors(unigram_thesaurus_dir)
    else:
        logging.warning('Skipping unigrams stage. Assuming output is at %s',
                        _find_events_file(unigram_thesaurus_dir))

    # FEATURE REWEIGHTING- will always be performed
    if 'ppmi' in stages:
        _do_ppmi(_find_events_file(unigram_thesaurus_dir),
                 unigram_thesaurus_dir_ppmi)

    # REDUCE DIMENSIONALITY
    # add in observed AN/NN vectors for SVD processing. Reduce both unigram vectors and observed phrase vectors
    # together and put the output into the same file
    unreduced_unigram_events_file = _find_events_file(unigram_thesaurus_dir_ppmi)
    # ...exp6-12/exp6.events.filtered.strings --> ...exp6-12/exp6
    reduced_file_prefix = join(unigram_thesaurus_dir_ppmi,
                               'exp%d-with-obs-phrases' % corpus)
    # only keep the most frequent types per PoS tag to speed things up
    counts = [('N', 200000), ('V', 200000), ('J', 100000), ('RB', 0), ('AN', 0), ('NN', 0)]
    if 'svd' in stages:
        if features_name == 'deps':
            # havent got observed vectors for these, do SVD on the unigrams only
            do_svd(unreduced_unigram_events_file, reduced_file_prefix,
                   desired_counts_per_feature_type=counts, reduce_to=[SVD_DIMS],
                   write=1)
        else:
            # in this case the name exp%d-with-obs-phrases is massively misleading because
            # there aren't any obs phrase vectors
            # let's just do SVD on the unigram phrases so we can compose them simply later
            do_svd(unreduced_unigram_events_file, reduced_file_prefix,
                   desired_counts_per_feature_type=counts, reduce_to=[SVD_DIMS],
                   apply_to=observed_phrasal_vectors)
    else:
        logging.warning('Skipping SVD stage. Assuming output is at %s-SVD*', reduced_file_prefix)

    # construct the names of files output by do_svd
    all_reduced_vectors = '%s-SVD%d.events.filtered.strings' % (reduced_file_prefix, SVD_DIMS)

    # TRAIN BARONI COMPOSER
    baroni_root_dir = join(prefix, 'baroni_guevara')
    trained_composer_file_bar = join(baroni_root_dir,
                                     '%s-SVD%s.baroni.pkl' % (basename(observed_phrasal_vectors), SVD_DIMS))
    trained_composer_file_guev = join(baroni_root_dir,
                                      '%s-SVD%s.guev.pkl' % (basename(observed_phrasal_vectors), SVD_DIMS))
    trained_composer_file_gref = join(prefix, 'gref_multistep', 'svo_comp.pkl')

    if 'baroni' in stages and features_name != 'deps':
        # todo use a Vectors object instead of a path to one as the first param to save time
        train_baroni_guevara_composers(all_reduced_vectors,
                                       baroni_root_dir,
                                       trained_composer_file_bar,
                                       trained_composer_file_guev,
                                       baroni_threshold=50)


    else:
        logging.warning('Skipping Baroni training stage. Assuming trained models are at: \n'
                        '\t\t\tBaroni:%s\n\t\t\tGuevara:%s', trained_composer_file_bar, trained_composer_file_guev)

    if 'gref' in stages and features_name != 'deps':
        train_grefenstette_multistep_composer(all_reduced_vectors,
                                              join(prefix, 'gref_multistep'))

    if 'compose' in stages:
        # it is OK for the first parameter to contain phrase vectors, there is explicit filtering coming up
        # the assumption is these are actually observed phrasal vectors
        compose_and_write_vectors(all_reduced_vectors,
                                  '%s-%s' % (dataset_name, SVD_DIMS),
                                  composer_algos,
                                  pretrained_Baroni_composer_file=trained_composer_file_bar,
                                  pretrained_Guevara_composer_file=trained_composer_file_guev,
                                  pretrained_Gref_composer_file=trained_composer_file_gref,
                                  output_dir=ngram_vectors_dir, dense_hd5=True)
    else:
        logging.warning('Skipping composition stage. Assuming output is at %s', ngram_vectors_dir)


def get_corpus_features_cmd_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--corpus', choices=('wikipedia', 'gigaword'), required=True,
                        help='Unlabelled corpus to source unigram vectors from')
    parser.add_argument('--features', choices=('dependencies', 'windows'), required=True,
                        help='Feature type of unigram vectors')
    return parser


def get_cmd_parser():
    parser = argparse.ArgumentParser(parents=[get_corpus_features_cmd_parser()])
    # add options specific to this script here
    parser.add_argument('--stages', choices=('unigrams', 'ppmi', 'svd',
                                             'baroni', 'gref', 'compose'),
                        required=True,
                        nargs='+',
                        help='What parts of the pipeline to run. Each part is independent, the pipeline can be '
                             'run incrementally. The stages are: '
                             '1) unigrams: extract unigram vectors from unlabelled corpus '
                             '2) ppmi: perform PPMI reweighting on feature counts '
                             '3) svd: reduce noun and adj matrix, apply to NP matrix '
                             '4) baroni: train Baroni composer '
                             '5) gref: train Grefenstette multistep SVO composer'
                             '6) compose: compose all possible NP vectors with all composers ')
    parser.add_argument('--use-ppmi', action='store_true',
                        help='If set, PPMI will be performed. Currently this is only implemented without SVD')
    return parser


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s")
    parameters = get_cmd_parser().parse_args()
    logging.info(parameters)

    corpus_name = parameters.corpus[:5]  # human-readable corpus name
    corpus = {'gigaword': 10, 'wikipedia': 11}
    features = {'dependencies': 12, 'windows': 13}
    logging.info('Starting pipeline with PPMI, SVD and Baroni composer')
    build_full_composed_thesauri_with_baroni_and_svd(corpus[parameters.corpus], features[parameters.features],
                                                     parameters.stages)
