import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
from sklearn.feature_selection.univariate_selection import _clean_nans
from discoutils.thesaurus_loader import Vectors
from thesisgenerator.utils.misc import calculate_log_odds, update_dict_according_to_mask

__author__ = 'mmb28'


class VectorBackedSelectKBest(SelectKBest):
    """
    An extention of sklearn's SelectKBest, which also contains a VectorStore. Feature selection is done
    in two optional steps:
        1: Remove all features that are not contained in the vector store
        2: Remove any remaining low-scoring features to ensure a maximum of k features are left fit

     Additionally, this class stores a vocabulary (like a vectorizer), which maps features to a corresponding columns
     in the feature vector matrix. This is so that a FeatureVectorsCsvDumper can be placed after this object in a
     pipeline.

     Also, this object assumes its input is not a matrix X (as in SelectKBest), but a tuple (X, vocabulary). The
     vocabulary is provided by ThesaurusVectorizer, which comes before this object in a pipeline and represents the
     mapping of features to columns in X before any feature selection is done.
    """

    def __init__(self, score_func=chi2, k='all', must_be_in_thesaurus=False, min_log_odds_score=0, **kwargs):
        """
        :param min_log_odds_score: any feature with a log odds score between -min_log_odds_score and
        min_log_odds_score will be removed. Assumes the classification problem is binary.
        """
        if not score_func:
            score_func = chi2
        self.k = k
        self.must_be_in_thesaurus = must_be_in_thesaurus
        self.min_log_odds_score = min_log_odds_score
        self.vocabulary_ = None
        super(VectorBackedSelectKBest, self).__init__(score_func=score_func, k=k)

    def fit(self, X, y, vector_source=None, clusters=None, **kwargs):
        if vector_source is None and clusters is None and self.must_be_in_thesaurus:
            logging.error('You requested feature selection based on vector presence '
                          'but did not provide a vector source.')
            raise ValueError('sector source (vectors or clusters) required with must_be_in_thesaurus')
        if self.must_be_in_thesaurus:
            self.vector_source = vector_source if vector_source else set(clusters.index)

        # Vectorizer also returns its vocabulary, store it and work with the rest
        X, self.vocabulary_ = X

        if self.k == 'all' or int(self.k) >= X.shape[1]:
            # do not bother calculating feature informativeness if all features will be used anyway
            self.scores_ = np.ones((X.shape[1],))
        else:
            super(VectorBackedSelectKBest, self).fit(X, y)

        self.vectors_mask = self._zero_score_of_oot_feats() \
            if self.must_be_in_thesaurus else np.ones(X.shape[1], dtype=bool)
        self.log_odds_mask = self._zero_score_of_low_log_odds_features(X, y) \
            if self.min_log_odds_score > 0 else np.ones(X.shape[1], dtype=bool);

        return self

    def transform(self, X):
        # Vectorizer also returns its vocabulary, remove it
        if self.vocabulary_:
            return super(VectorBackedSelectKBest, self).transform(X[0]), self.vocabulary_
        else:
            # Sometimes the training set contain no features. We don't want this to break the experiment,
            # so let is slide
            logging.error('Empty vocabulary')
            return X[0], self.vocabulary_

    def _zero_score_of_oot_feats(self):
        mask = np.ones(self.scores_.shape, dtype=bool)
        for feature, index in self.vocabulary_.items():
            if feature not in self.vector_source:
                mask[index] = False
        if np.count_nonzero(mask) == 0:
            logging.error('Feature selector removed all features')
            raise ValueError('Empty vocabulary')
        return mask

    def _zero_score_of_low_log_odds_features(self, X, y):
        if self.min_log_odds_score <= 0:
            # we don't want to use log odds score, return an all-true mask
            return np.ones(X.shape[1])
        if len(set(y)) != 2:
            raise ValueError('Calculating a log odds score requires a binary classification task')
        log_odds = calculate_log_odds(X, y)
        return (log_odds > self.min_log_odds_score) | (log_odds < -self.min_log_odds_score)

    def _get_support_mask(self):
        k = self.k
        chi2_scores = self.scores_
        chi2_mask = np.ones(chi2_scores.shape, dtype=bool)

        if k != 'all' and k < len(chi2_scores):
            # we don't want all features to be kept, and the number we want is less than the number available
            chi2_scores = _clean_nans(chi2_scores)
            selected_indices = np.argsort(chi2_scores)[:k]
            chi2_mask[selected_indices] = False

        mask = chi2_mask & self.vectors_mask & self.log_odds_mask
        logging.info('%d/%d features survived feature selection', np.count_nonzero(mask), len(mask))

        # Only keep the scores of the features that survived. This array is used to check the
        # input data shape at train and decode time matches. However, because the post-feature-selections
        # vocabulary is passed back into the vectorizer, at decode time the input will likely be smaller. This is
        # like doing feature selection in the vectorizer.
        self.scores_ = self.scores_[mask]
        self.log_odds_mask = self.log_odds_mask[mask]
        self.vectors_mask = self.vectors_mask[mask]

        self.vocabulary_ = update_dict_according_to_mask(self.vocabulary_, mask)
        return mask


class MetadataStripper(BaseEstimator, TransformerMixin):
    """
    The current implementation of ThesaurusVectorizer's fit() returns not just a data matrix, but also some
    metadata (its vocabulary). This class is meant to sit in a pipeline behind the vectorizer to remove that
    metadata, so that it doesn't break other items in the pipeline.

    Currently several other pipeline elements can make use of this data ( VectorBackedSelectKBest and
     FeatureVectorsCsvDumper). This class must come after these in a pipeline as they do not have any
     defensive checks
    """

    def fit(self, X, y, vector_source=None, strategy='linear', k=None, **kwargs):
        matrix, self.voc = X  # store voc, may be handy for for debugging
        self.vector_source = vector_source
        if isinstance(self.vector_source, Vectors):
            # the vector source can be either a Thesaurus or Vectors. Both can provide nearest neighbours,
            # but the latter needs this method to be called first
            if not k:
                k = 10
            self.vector_source.init_sims([str(foo) for foo in self.voc.keys()],
                                         strategy=strategy, n_neighbors=k)
        return self

    def transform(self, X, **kwargs):
        # if X is a tuple, strip metadata, otherwise let it be
        return X[0] if tuple(X) == X else X

    def get_params(self, deep=True):
        return super(MetadataStripper, self).get_params(deep)
