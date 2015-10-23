import os
import sys
from os.path import join

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
import argparse
import logging
import pandas as pd
from discoutils.cmd_utils import run_and_log_output
from discoutils.misc import temp_chdir, mkdirs_if_not_exists
from discoutils.io_utils import write_vectors_to_hdf
from discoutils.tokens import DocumentFeature
from builder.composers.vectorstore import (RightmostWordComposer, LeftmostWordComposer,
                                           MultiplicativeComposer, AdditiveComposer,
                                           VerbComposer, compose_and_write_vectors)


def run_glove():
    logging.info('Starting training')
    with temp_chdir(args.glove_dir):
        run_and_log_output('sh {} {}'.format(glove_script, unlabelled_data))

    # convert their format to ours
    df = pd.read_csv(raw_vectors_file, sep=' ', index_col=0, header=None)
    logging.info('Done training, filtering junk and converting %d vectors to Byblo-compatible format', len(df))
    # remove any shit-looking tokens, they'll get in the way later
    mask = [DocumentFeature.from_string(x).type != 'EMPTY' and 3 < len(x) < 20 for x in df.index]
    logging.info('Keeping %d entries', sum(mask))
    logging.info('Shape of vectors before filtering %r', df.shape)
    df = df[mask]
    logging.info('Shape of vectors after filtering %r', df.shape)
    cols = ['f%d' % i for i in range(df.shape[1])]
    mkdirs_if_not_exists(output_dir)
    write_vectors_to_hdf(df.values, df.index, cols, formatted_vectors_file)


def reformat_data():
    # glove requires the entire corpus to be on a single row
    logging.info('Starting corpus reformat')
    run_and_log_output('cat {}/* > tmp'.format(pos_only_data_dir))
    run_and_log_output('tr "\\n" " " < tmp > {}'.format(unlabelled_data))
    run_and_log_output('rm -f tmp')
    logging.info('Done with reformat')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--stages', choices=('reformat', 'vectors', 'compose'),
                        required=True, nargs='+',
                        help="""Stages of pipeline to run. These include:
                        - reformat: convert data to the right format
                        - vectors: run glove embedding code
                        - compose: build phrase representations too""")
    parser.add_argument('--glove-dir', required=True, help='Path to downloaded glove directory')
    parser.add_argument('--corpus', choices=('gigaw', 'wiki'), required=True,
                        help='Name of unlabelled corpus to train embeddings on')
    args = parser.parse_args()

    prefix = os.path.abspath(os.path.join(__file__, '..', '..'))
    output_dir = join(prefix, 'outputs', 'glove')

    # BUILD UP PATHS TO INPUTS AND OUTPUTS
    glove_script = join(args.glove_dir, 'demo2.sh')  # TODO explain how to set param in that script

    if args.corpus == 'wiki':
        # todo explain what these are and why formatting is needed
        pos_only_data_dir = join(prefix, 'data/wiki/')
        unlabelled_data = join(prefix, 'data/wikipedia.oneline')
    else:
        pos_only_data_dir = join(prefix, 'data/gigaw/')
        unlabelled_data = join(prefix, 'data/gigaw.oneline')

    raw_vectors_file = join(args.glove_dir, 'vectors.txt')  # what GloVe produces
    formatted_vectors_file = join(output_dir, 'vectors.%s.h5' % args.corpus)  # unigram vectors in my format

    # DO THE ACTUAL WORK
    composer_algos = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer,
                      RightmostWordComposer, VerbComposer]

    if 'reformat' in args.stages:
        reformat_data()

    if 'vectors' in args.stages:
        run_glove()

    if 'compose' in args.stages:
        logging.info('Loading labelled corpora and composing phrase vectors therein')
        compose_and_write_vectors(formatted_vectors_file,
                                  'glove-%s'%args.corpus,
                                  composer_algos,
                                  output_dir=output_dir,
                                  dense_hd5=True)
