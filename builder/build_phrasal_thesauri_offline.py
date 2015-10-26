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
                                  unindex_all_byblo_vectors)
from discoutils.reweighting import ppmi_sparse_matrix
from discoutils.reduce_dimensionality import do_svd
from discoutils.misc import temp_chdir
from discoutils.thesaurus_loader import Vectors
from builder.composers.vectorstore import (AdditiveComposer, MultiplicativeComposer,
                                           LeftmostWordComposer, RightmostWordComposer,
                                           compose_and_write_vectors, VerbComposer)


def calculate_unigram_vectors(byblo_conf_file, byblo_base_dir):
    # find out where the conf file said output should go
    opts, _ = parse_byblo_conf_file(byblo_conf_file)
    byblo_output_prefix = join(opts.output, basename(opts.input))

    # get byblo to calculate vectors for all entries
    set_stage_in_byblo_conf_file(byblo_conf_file, 1)
    with temp_chdir(byblo_base_dir):
        run_byblo(byblo_conf_file)
        set_stage_in_byblo_conf_file(byblo_conf_file, 0)
        # get vectors as strings
        unindex_all_byblo_vectors(byblo_output_prefix)


def _find_conf_file(thesaurus_dir):
    try:
        return glob(join(thesaurus_dir, '*conf*'))[0]
    except IndexError:
        raise ValueError('No Byblo conf file in %s' % thesaurus_dir)


def _find_events_file(thesaurus_dir):
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


# def build_full_composed_thesauri_with_baroni_and_svd(corpus, corpus_name, stages):
def build_full_composed_thesauri_with_baroni_and_svd(args):
    # SET UP A FEW REQUIRED PATHS

    byblo_opts, _ = parse_byblo_conf_file(args.conf)
    input_file_name = os.path.basename(byblo_opts.input)
    # INPUT 1:  DIRECTORY. Must contain a single conf file
    unigram_vectors_dir = os.path.abspath(byblo_opts.output)
    mkdirs_if_not_exists(unigram_vectors_dir)
    unigram_vectors_dir_ppmi = '%s-ppmi' % os.path.dirname(byblo_opts.output)
    mkdirs_if_not_exists(unigram_vectors_dir_ppmi)
    unigram_vectors_dir_ppmi_svd = '%s-ppmi-svd' % os.path.dirname(byblo_opts.output)
    mkdirs_if_not_exists(unigram_vectors_dir_ppmi_svd)

    # INPUT 2: A FILE, TSV, underscore-separated observed vectors for ANs and NNs
    SVD_DIMS = 100

    ngram_vectors_dir = '%s-ppmi-svd-composed' % os.path.dirname(byblo_opts.output)
    mkdirs_if_not_exists(ngram_vectors_dir)
    composer_algos = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer,
                      VerbComposer, RightmostWordComposer]

    # EXTRACT UNIGRAM VECTORS WITH BYBLO
    if 'unigrams' in args.stages:
        calculate_unigram_vectors(os.path.abspath(args.conf), os.path.abspath(args.byblo))
    else:
        logging.warning('Skipping unigrams stage. Assuming output is at %s',
                        byblo_opts.output)

    # FEATURE REWEIGHTING- will always be performed
    if 'ppmi' in args.stages:
        _do_ppmi(_find_events_file(byblo_opts.output), unigram_vectors_dir_ppmi)

    # REDUCE DIMENSIONALITY
    # add in observed AN/NN vectors for SVD processing. Reduce both unigram vectors and observed phrase vectors
    # together and put the output into the same file
    unreduced_unigram_events_file = _find_events_file(unigram_vectors_dir_ppmi)
    # ...exp6-12/exp6.events.filtered.strings --> ...exp6-12/exp6
    reduced_file_prefix = join(unigram_vectors_dir_ppmi_svd, input_file_name)
    # only keep the most frequent types per PoS tag to speed things up
    counts = [('N', 200000), ('V', 200000), ('J', 100000), ('RB', 0), ('AN', 0), ('NN', 0)]
    if 'svd' in args.stages:
        # in this case the name exp%d-with-obs-phrases is massively misleading because
        # there aren't any obs phrase vectors
        # let's just do SVD on the unigram phrases so we can compose them simply later
        do_svd(unreduced_unigram_events_file, reduced_file_prefix,
               desired_counts_per_feature_type=counts, reduce_to=[SVD_DIMS])
    else:
        logging.warning('Skipping SVD stage. Assuming output is at %s-SVD*', reduced_file_prefix)

    # construct the names of files output by do_svd
    all_reduced_vectors = '%s-SVD%d.events.filtered.strings' % (reduced_file_prefix, SVD_DIMS)

    if 'compose' in args.stages:
        # it is OK for the first parameter to contain phrase vectors, there is explicit filtering coming up
        # the assumption is these are actually observed phrasal vectors
        compose_and_write_vectors(all_reduced_vectors,
                                  '%s-%s' % (input_file_name, SVD_DIMS),
                                  composer_algos, output_dir=ngram_vectors_dir, dense_hd5=True)
    else:
        logging.warning('Skipping composition stage. Assuming output is at %s', ngram_vectors_dir)


def get_cmd_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--conf', required=True, help='Path to Byblo conf file')
    parser.add_argument('--byblo', required=True, help='Path to Byblo executable')
    parser.add_argument('--stages', choices=('unigrams', 'ppmi', 'svd',
                                             'baroni', 'compose'),
                        required=True,
                        nargs='+',
                        help='What parts of the pipeline to run. Each part is independent, the pipeline can be '
                             'run incrementally. The stages are: '
                             '1) unigrams: extract unigram vectors from unlabelled corpus '
                             '2) ppmi: perform PPMI reweighting on feature counts '
                             '3) svd: reduce noun and adj matrix, apply to NP matrix '
                             '4) compose: compose all possible NP vectors with all composers ')
    parser.add_argument('--use-ppmi', action='store_true',
                        help='If set, PPMI will be performed. Currently this is only implemented without SVD')
    return parser


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s")
    parameters = get_cmd_parser().parse_args()
    logging.info(parameters)
    logging.info('Starting pipeline with PPMI, SVD and Baroni composer')
    build_full_composed_thesauri_with_baroni_and_svd(parameters)
