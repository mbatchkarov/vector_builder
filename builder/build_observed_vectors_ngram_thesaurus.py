import argparse
import logging
import sys
import os

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from discoutils.cmd_utils import run_and_log_output
from thesisgenerator.utils.misc import force_symlink


'''
If using SVD, symlink the reduced vectors for all unigrams and NPs (done by build_phrasal_..
as as part of training Baroni) to the right location.
Otherwise add unigram vectors (must exists) to ngram observed vectors (must exist)
and write to a single file in e.g. exp10-13-composed-vectors
'''


def do_work(corpus, features, svd_dims):
    prefix = '/lustre/scratch/inf/mmb28/FeatureExtractionToolkit'
    name = 'wiki' if corpus == 11 else 'gigaw'

    # where should output be written
    svd_appendage = '' if svd_dims == 0 else '-%d' % svd_dims
    output_file = os.path.join(prefix,
                               'exp%d-%d-composed-ngrams-ppmi-svd' % (corpus, features),
                               'AN_NN_%s%s_Observed.events.filtered.strings' % (name, svd_appendage))

    # contains SVD-reduced N,J and NP observed vectors, built by other script
    vectors_file = '%s/exp%d-%db/exp%d-with-obs-phrases-SVD%d.events.filtered.strings' % \
                                  (prefix, corpus, features, corpus, svd_dims)
    force_symlink(vectors_file, output_file)



def get_cmd_parser():
    from thesisgenerator.scripts.build_phrasal_thesauri_offline import get_corpus_features_cmd_parser

    parser = argparse.ArgumentParser(parents=[get_corpus_features_cmd_parser()])
    # add options specific to this script here
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--svd', choices=(0, 30, 100, 300, 1000), nargs='+', type=int,
                       help='What SVD dimensionalities to build observed-vector thesauri from. '
                            'Vectors must have been produced and reduced already. 0 stand for unreduced.')
    return parser


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s")

    parameters = get_cmd_parser().parse_args()
    logging.info(parameters)

    corpus = 10 if parameters.corpus == 'gigaword' else 11
    if parameters.features == 'dependencies':
        raise ValueError('Observed dependency vectors for NPs do not exist')
    features = 13
    for dims in parameters.svd:
        do_work(corpus, features, dims)
