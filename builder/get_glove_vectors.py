import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
import argparse
import logging
from os.path import join
import pandas as pd
from discoutils.cmd_utils import run_and_log_output
from discoutils.misc import temp_chdir
from discoutils.io_utils import write_vectors_to_hdf
from discoutils.tokens import DocumentFeature
from thesisgenerator.composers.vectorstore import (RightmostWordComposer, LeftmostWordComposer,
                                                   MultiplicativeComposer, AdditiveComposer,
                                                   VerbComposer, compose_and_write_vectors)


prefix = '/lustre/scratch/inf/mmb28/FeatureExtractionToolkit'
glove_dir = join(prefix, 'glove')
glove_script = join(glove_dir, 'demo_miro.sh')  # set param in that script
pos_only_data_dir = join(prefix, 'data/wikipedia-tagged-pos/wikipedia/')
unlabelled_data = join(prefix, 'data/wikipedia-tagged-pos/wikipedia.oneline')
raw_vectors_file = join(glove_dir, 'vectors.txt')  # what GloVe produces
formatted_vectors_file = join(glove_dir, 'vectors.miro.h5')  # unigram vectors in my format
composer_algos = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer,
                  RightmostWordComposer, VerbComposer]


def compute_and_write_vectors(stages):
    if 'reformat' in stages:
        # glove requires the entire corpus to be on a single row
        logging.info('Starting corpus reformat')
        run_and_log_output('cat {}/* > tmp'.format(pos_only_data_dir))
        run_and_log_output('tr "\\n" " " < tmp > {}'.format(unlabelled_data))
        run_and_log_output('rm -f tmp')
        logging.info('Done with reformat')

    if 'vectors' in stages:
        logging.info('Starting training')
        with temp_chdir(glove_dir):
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
        write_vectors_to_hdf(df.values, df.index, cols, formatted_vectors_file)

    if 'compose' in stages:
        logging.info('Loading labelled corpora and composing phrase vectors therein')
        compose_and_write_vectors(formatted_vectors_file,
                                  'glove-wiki',
                                  composer_algos,
                                  output_dir=glove_dir,
                                  dense_hd5=True)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', choices=('reformat', 'vectors', 'compose'),
                        required=True, nargs='+')
    args = parser.parse_args()
    compute_and_write_vectors(args.stages)

