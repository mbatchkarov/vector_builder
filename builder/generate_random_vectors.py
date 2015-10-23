import argparse
import logging

import numpy as np
import pandas as pd

from eval.scripts.compress_labelled_data import get_all_document_features
from discoutils.thesaurus_loader import DenseVectors

"""
Generates a random vector for each NP in all labelled corpora
"""

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
parser = argparse.ArgumentParser(description='Generate a random vectors for a set of words')
parser.add_argument('--output', required=True, help='Path to output file')
parser.add_argument('--dim', type=int, default=100, help='Dimensionality of vectors')

args = parser.parse_args()

np.random.seed(0)
feats = ['rand%d' % i for i in range(args.dim)]
phrases = list(get_all_document_features(include_unigrams=True))
vectors = np.random.random((len(phrases), args.dim))

v = DenseVectors(pd.DataFrame(vectors, index=phrases, columns=feats))
v.to_tsv(args.output, dense_hd5=True)
