from functools import reduce
import logging
import os
from random import sample
from pickle import load
from copy import deepcopy

import numpy as np
import scipy.sparse as sp
import pandas as pd
import six
from composes.composition.lexical_function import LexicalFunction
from discoutils.io_utils import write_vectors_to_disk, write_vectors_to_hdf
from discoutils.thesaurus_loader import Thesaurus, Vectors
from discoutils.tokens import DocumentFeature


def check_vectors(unigram_source):
    if not unigram_source:
        raise ValueError('Composers need a unigram vector source')
    if not hasattr(unigram_source, 'get_vector'):
        raise ValueError('Creating a composer requires a Vectors data structure that holds unigram vectors')
    return unigram_source


class ComposerMixin(object):
    def compose_all(self, phrases):
        """
        Composes all `phrases` and returns all unigrams and `phrases` as a matrix. Does NOT store the composed vectors.
        Unigram vectors must be brought in by extending classes.
        :param phrases: iterable of `str` or `DocumentFeature`
        :return: a tuple of :
            1) `csr_matrix` containing all vectors, unigram and composed
            2) the columns (features) of the unigram space that was used for composition
            3) a row index- dict {Feature: Row}. Maps from a feature to the row in 1) where the vector for that
               feature is. Note: This is the opposite of what IO functions in discoutils expect
        """
        composable_phrases = [foo for foo in phrases if foo in self]
        logging.info('Composing... %s able to compose %d/%d phrases using %d unigrams',
                     self.name, len(composable_phrases), len(phrases), len(self.unigram_source.name2row))
        if not composable_phrases:
            raise ValueError('%s cannot compose any of the provided phrases' % self.name)
        new_matrix = sp.vstack(self.get_vector(foo) for foo in composable_phrases)
        old_len = len(self.unigram_source.name2row)
        all_rows = deepcopy(self.unigram_source.name2row)  # can't mutate the unigram datastructure
        for i, phrase in enumerate(composable_phrases):
            key = phrase if isinstance(phrase, str) else str(phrase)
            # phrase shouln't be in the unigram source.
            assert key not in all_rows
            all_rows[key] = i + old_len  # this will not append to all_rows if phrase is contained in unigram_source
        all_vectors = sp.vstack([self.unigram_source.matrix, new_matrix], format='csr')
        assert all_vectors.shape == (len(all_rows), len(self.unigram_source.columns)), 'Shape mismatch'
        return all_vectors, self.unigram_source.columns, all_rows


class AdditiveComposer(Vectors, ComposerMixin):
    name = 'Add'
    # composers in general work with n-grams (for simplicity n<4)
    entry_types = {'2-GRAM', '3-GRAM', 'AN', 'NN', 'VO', 'SVO'}

    def __init__(self, unigram_source):
        self.unigram_source = check_vectors(unigram_source)
        self.function = np.add

    def get_vector(self, feature):
        """
        :type feature: DocumentFeature
        :rtype: scipy.sparse.csr_matrix
        """
        if isinstance(feature, six.string_types):
            feature = DocumentFeature.from_string(feature)
        return sp.csr_matrix(reduce(self.function,
                                    [self.unigram_source.get_vector(str(t)).A for t in feature[:]]))

    def contains_impl(self, feature):
        """
        Contains all sequences of words where we have a distrib vector for each unigram
        they contain. Rejects unigrams.
        """
        # if isinstance(feature, six.string_types):
        #     feature = DocumentFeature.from_string(feature)

        feat_str = str(feature) if isinstance(feature, DocumentFeature) else feature
        feat_df = feature if isinstance(feature, DocumentFeature) else DocumentFeature.from_string(feature)

        if feat_df.type not in self.entry_types:
            # no point in trying
            return False
        return all(f in self.unigram_source for f in feat_str.split(DocumentFeature.ngram_separator))

    def __contains__(self, feature):
        return self.contains_impl(feature)

    def __str__(self):
        return '[%s with %d unigram entries]' % (self.__class__.__name__, len(self.unigram_source))

    def __len__(self):
        # this will also get call when __nonzero__ is called
        return len(self.unigram_source)


class AverageComposer(AdditiveComposer):
    name = 'Avg'
    entry_types = {'2-GRAM', '3-GRAM', 'AN', 'NN', 'VO', 'SVO'}

    def __init__(self, unigram_source):
        self.unigram_source = check_vectors(unigram_source)
        self.function = np.add

    def get_vector(self, feature):
        v = super().get_vector(feature)  # Add
        return v / 2


class MultiplicativeComposer(AdditiveComposer):
    name = 'Mult'

    def __init__(self, unigram_source):
        self.unigram_source = check_vectors(unigram_source)
        self.function = np.multiply


class MinComposer(MultiplicativeComposer):
    name = 'Min'

    def __init__(self, unigram_source):
        self.unigram_source = check_vectors(unigram_source)
        self.function = lambda m, n: np.minimum(m, n)


class MaxComposer(MinComposer):
    name = 'Max'

    def __init__(self, unigram_source):
        self.unigram_source = check_vectors(unigram_source)
        self.function = lambda m, n: np.maximum(m, n)


class LeftmostWordComposer(AdditiveComposer):
    name = 'Left'
    entry_types = {'2-GRAM', '3-GRAM', 'AN', 'NN', 'VO', 'SVO'}

    def __init__(self, unigram_source):
        self.unigram_source = check_vectors(unigram_source)
        self.hardcoded_index = 0

    def get_vector(self, feature):
        if isinstance(feature, six.string_types):
            feature = DocumentFeature.from_string(feature)
        return self.unigram_source.get_vector(str(feature[self.hardcoded_index]))

    def contains_impl(self, feature):
        if isinstance(feature, six.string_types):
            feature = DocumentFeature.from_string(feature)
        if feature.type not in self.entry_types:
            # no point in composing single-word document features
            return False
        return str(feature[self.hardcoded_index]) in self.unigram_source


class RightmostWordComposer(LeftmostWordComposer):
    name = 'Right'

    def __init__(self, unigram_source):
        self.unigram_source = check_vectors(unigram_source)
        self.hardcoded_index = -1


class VerbComposer(LeftmostWordComposer):
    """
    Represents verb phrases by the vector of their head
    """
    name = 'Verb'
    entry_types = {'SVO'}

    def __init__(self, unigram_source):
        self.unigram_source = check_vectors(unigram_source)
        self.hardcoded_index = 1


class BaroniComposer(Vectors, ComposerMixin):
    entry_types = {'AN', 'NN'}
    name = 'Baroni'

    def __init__(self, unigram_source, pretrained_model_file):
        self.unigram_source = check_vectors(unigram_source)
        if not pretrained_model_file:
            logging.error('Expected filename, got %s', pretrained_model_file)
            raise ValueError('Model file required to perform composition.')
        with open(pretrained_model_file, 'rb') as infile:
            self._composer = load(infile)

        # verify the composer's internal structure matches the unigram source
        self.available_modifiers = set(self._composer.function_space.id2row)

        core_space = self.unigram_source.to_dissect_core_space()
        assert list(unigram_source.columns) == (self._composer.composed_id2column)
        self.dissect_core_space = core_space

        # check composed space's columns matches core space's (=unigram source)'s columns
        assert core_space.id2column == self._composer.composed_id2column

    def __contains__(self, feature):
        """
        Accept all adjective-noun or noun-noun phrases where we have a corpus-observed vector for the head and
        a learnt matrix (through PLSR) for the modifier
        """
        # todo expand unit tests now that we have a real composer
        if feature.type not in self.entry_types:
            # ignore non-AN features
            return False

        modifier, head = feature.tokens
        assert ('J', 'N') == (modifier.pos, head.pos) or ('N', 'N') == (modifier.pos, head.pos)

        # if DocumentFeature('1-GRAM', (noun,)) not in self.unigram_source:
        if DocumentFeature.from_string(str(head)) not in self.unigram_source:
            # ignore ANs containing unknown nouns
            return False

        # ignore ANs containing unknown adjectives
        return str(modifier) in self.available_modifiers

    def __str__(self):
        return '[BaroniComposer with %d modifiers and %d heads]' % \
               (len(self.available_modifiers), len(self.unigram_source))

    def __repr__(self):
        return str(self)

    def __len__(self):
        # this will also get call when __nonzero__ is called
        return len(self.available_modifiers)

    def get_vector(self, feature):
        # todo test properly
        """

        :param feature: DocumentFeature to compose, assumed to be an adjective/noun and a noun, with PoS tags
        :return:
         :rtype: 1xN scipy sparse matrix of type numpy.float64 with M stored elements in Compressed Sparse Row format,
         where N is the dimensionality of the vectors in the unigram source
        """
        modifier = str(feature.tokens[0])
        head = str(feature.tokens[1])
        phrase = '{}_{}'.format(modifier, head)
        x = self._composer.compose([(modifier, head, phrase)],
                                   self.dissect_core_space).cooccurrence_matrix.mat
        return x


class GuevaraComposer(BaroniComposer):
    entry_types = {'AN', 'NN'}
    name = 'Guevara'

    def __init__(self, unigram_source, pretrained_model_file, *args):
        self.unigram_source = check_vectors(unigram_source)
        if not pretrained_model_file:
            logging.error('Expected filename, got %s', pretrained_model_file)
            raise ValueError('Model file required to perform composition.')
        with open(pretrained_model_file, 'rb') as infile:
            self._composer = load(infile)

        assert list(unigram_source.columns) == list(self._composer.composed_id2column)
        self.dissect_core_space = self.unigram_source.to_dissect_core_space()

        # check composed space's columns matches core space's (=unigram source)'s columns
        assert self.dissect_core_space.id2column == self._composer.composed_id2column

    def __str__(self):
        return '[GuevaraComposer with %d modifiers and %d heads]' % \
               (len(self.available_modifiers), len(self.unigram_source))

    def __contains__(self, feature):
        # both head and modifier need to have unigram vectors.
        # I don't see why the modifier needs a vector, given that we're using
        # its matrix representation instead, but that is what dissect does
        if isinstance(feature, six.string_types):
            feature = DocumentFeature.from_string(feature)

        if feature.type not in self.entry_types:
            # no point in trying
            return False
        return all(str(f) in self.unigram_source for f in feature[:])


class GrefenstetteMultistepComposer(BaroniComposer):
    entry_types = {'SVO'}
    name = 'Multistep'

    def __init__(self, unigram_source, v_model):
        self.unigram_source = check_vectors(unigram_source)
        self.n_space = self.unigram_source.to_dissect_core_space()
        with open(v_model, 'rb') as infile:
            self.v_model = load(infile)
            # with open(vo_model, 'rb') as infile:
            # self.vo_model = load(infile)
        self.verbs = self.v_model.function_space.id2row
        logging.info('Multistep composer has these verbs:', self.verbs)

    def __str__(self):
        'Multistep composer with %d verbs and %d nouns' % (len(self.verbs),
                                                           len(self.unigram_source))

    def __contains__(self, feature):
        if isinstance(feature, six.string_types):
            feature = DocumentFeature.from_string(feature)

        # this is a SVO, we have a verb tensor and vectors for both arguments
        return feature.type in self.entry_types and \
               feature[1] in self.verbs and \
               feature[0] in self.unigram_source and \
               feature[2] in self.unigram_source
        # alternative- try to compose. if ValueError, we can't

    def get_vector(self, df):
        # 3. use the trained models to compose new SVO sentences
        # 3.1 use the V model to create new VO combinations
        data = (str(df[1]), str(df[2]), str(df[1:]))
        # ("take/V", "place/N", "take/V_place/N")
        vo_composed_space = self.v_model.compose([data], self.n_space)
        # todo how do we get VO vectors? these are (100x100)+100 dimensional (intercept).
        # todo do we allow document features of different dimensionality
        # vo_composed_space.cooccurrence_matrix.mat

        # 3.2 the new VO combinations will be used as functions:
        # load the new VO combinations obtained through composition into
        # a new composition model
        expanded_vo_model = LexicalFunction(function_space=vo_composed_space,
                                            intercept=self.v_model._has_intercept)

        # 3.3 use the new VO combinations by composing them with subject nouns
        # in order to obtain new SVO sentences
        data = (str(df[1:]), str(df[0]), str(df))
        svo_composed_space = expanded_vo_model.compose([data], self.n_space)

        # print the composed spaces:
        # logging.info("SVO composed space:")
        # logging.info(svo_composed_space.id2row)
        # logging.info(svo_composed_space.cooccurrence_matrix)

        # get vectors out. these are 100-dimensional
        return svo_composed_space.cooccurrence_matrix.mat


class CopyObject(Vectors, ComposerMixin):
    name = 'CopyObj'
    entry_types = {'SVO'}

    def __init__(self, verbs_file, unigram_source):
        self.verb_tensors = dict()
        with pd.get_store(verbs_file) as store:
            for verb in store.keys():
                self.verb_tensors[verb[1:] + '/V'] = store[verb].values
        logging.info('Found %d verb tensors in %s', len(self.verb_tensors), verbs_file)
        if not self.verb_tensors:
            raise ValueError('Cant build a categorical model without verb matrices')

        self.unigram_source = unigram_source

    def __contains__(self, feature):
        if isinstance(feature, six.string_types):
            feature = DocumentFeature.from_string(feature)

        # this is a SVO, we have a verb tensor and vectors for both arguments
        return feature.type in self.entry_types and \
               str(feature[1]) in self.verb_tensors and \
               str(feature[0]) in self.unigram_source and \
               str(feature[2]) in self.unigram_source

    def get_vector(self, phrase_df):
        subj, verb, obj = map(str, phrase_df.tokens)
        subj_v = self.unigram_source.get_vector(subj).A.T  # shape 100x1
        verb_m = self.verb_tensors[verb]  # shape 100x100
        obj_v = self.unigram_source.get_vector(obj).A.T  # shape 100x1

        vec = subj_v * np.dot(verb_m, obj_v)
        return sp.csr_matrix(vec.T)  # type needs to be compatible w other composers

    def __str__(self):
        return '%s composer with %d verbs and %d unigrams' % (self.name,
                                                              len(self.verb_tensors),
                                                              len(self.unigram_source))


class FrobeniusAdd(CopyObject):
    name = 'FAdd'
    entry_types = {'SVO'}
    function = np.add

    def get_vector(self, phrase_df):
        subj, verb, obj = map(str, phrase_df.tokens)
        subj_v = self.unigram_source.get_vector(subj).A.T  # shape 100x1
        verb_m = self.verb_tensors[verb]  # shape 100x100
        obj_v = self.unigram_source.get_vector(obj).A.T  # shape 100x1

        vec = self.function((subj_v * np.dot(verb_m, obj_v)), (obj_v * np.dot(verb_m.T, subj_v)))
        return sp.csr_matrix(vec.T)


class FrobeniusMult(FrobeniusAdd):
    name = 'FMult'
    entry_types = {'SVO'}
    function = np.multiply


class DummyThesaurus(Thesaurus):
    """
    A thesaurus-like object which return "b/N" as the only neighbour of every possible entry
    """
    name = 'Constant'

    def __init__(self):
        pass

    def get_nearest_neighbours(self, feature):
        return [('b/N', 1.0)]

    def get_vector(self):
        pass

    def to_shelf(self, *args, **kwargs):
        pass

    def __len__(self):
        return 9999999

    def __contains__(self, feature):
        return True


class RandomThesaurus(DummyThesaurus):
    """
    A thesaurus-like object which returns a single random neighbour for every possible entry. That neighbour
    is chosen from the vocabulary that is passed in (as a dict {feature:index} )
    """
    name = 'Random'

    def __init__(self, vocab=None, k=1):
        self.vocab = vocab
        self.k = k

    def get_nearest_neighbours(self, item):
        if not self.vocab:
            raise ValueError('You need to provide a set of value to choose from first.')
        return [(str(foo), 1.) for foo in sample(self.vocab, self.k)]


def _default_row_filter(feat_str: str, feat_df: DocumentFeature):
    return feat_df.tokens[0].pos in {'N', 'J', 'V'} and feat_df.type == '1-GRAM'


def compose_and_write_vectors(unigram_vectors, short_vector_dataset_name, composer_classes,
                              pretrained_Baroni_composer_file=None, pretrained_Guevara_composer_file=None,
                              pretrained_Gref_composer_file=None, categorical_vector_matrix_file=None,
                              output_dir='.', gzipped=True, dense_hd5=False,
                              row_filter=_default_row_filter):
    """
    Extracts all composable features from a labelled classification corpus and dumps a composed vector for each of them
    to disk. The output file will also contain all unigram vectors that were passed in, and only unigrams!
    :param unigram_vectors: a file in Byblo events format that contain vectors for all unigrams OR
    a Vectors object. This will be used in the composition process.
    :type unigram_vectors: str or Vectors
    :param classification_corpora: Corpora to extract features from. Dict {corpus_path: conf_file}
    :param pretrained_Baroni_composer_file: path to pre-trained Baroni AN/NN composer file
    :param output_dir:
    :param composer_classes: what composers to use
    :type composer_classes: list
    """
    from thesisgenerator.scripts.extract_NPs_from_labelled_data import get_all_NPs_VPs

    phrases_to_compose = get_all_NPs_VPs()
    # if this isn't a Vectors object assume it's the name of a file containing vectors and load them
    if not isinstance(unigram_vectors, Vectors):
        # ensure there's only unigrams in the set of unigram vectors
        # composers do not need any ngram vectors contain in this file, they may well be
        # observed ones
        unigram_vectors = Vectors.from_tsv(unigram_vectors,
                                           # todo enforce_word_entry_pos_format=False??? Why was that needed?
                                           row_filter=row_filter)
        logging.info('Starting composition with %d unigram vectors', len(unigram_vectors))

    # doing this loop in parallel isn't worth it as pickling or shelving `vectors` is so slow
    # it negates any gains from using multiple cores
    for composer_class in composer_classes:
        if composer_class == BaroniComposer:
            assert pretrained_Baroni_composer_file is not None
            composer = BaroniComposer(unigram_vectors, pretrained_Baroni_composer_file)
        elif composer_class == GuevaraComposer:
            assert pretrained_Guevara_composer_file is not None
            composer = GuevaraComposer(unigram_vectors, pretrained_Guevara_composer_file)
        elif composer_class == GrefenstetteMultistepComposer:
            assert pretrained_Gref_composer_file is not None
            composer = GrefenstetteMultistepComposer(unigram_vectors, pretrained_Gref_composer_file)
        elif composer_class in [CopyObject, FrobeniusAdd, FrobeniusMult]:
            composer = composer_class(categorical_vector_matrix_file, unigram_vectors)
        else:
            composer = composer_class(unigram_vectors)

        try:
            # compose_all returns all unigrams and composed phrases
            mat, cols, rows = composer.compose_all(phrases_to_compose)

            events_path = os.path.join(output_dir,  # todo name AN_NN no longer appropriate, whatever
                                       'AN_NN_%s_%s.events.filtered.strings' % (short_vector_dataset_name,
                                                                                composer.name))
            if dense_hd5:
                write_vectors_to_hdf(mat, rows, cols, events_path)
            else:
                rows2idx = {i: DocumentFeature.from_string(x) for (x, i) in rows.items()}
                write_vectors_to_disk(mat.tocoo(), rows2idx, cols, events_path,
                                      entry_filter=lambda x: x.type in {'AN', 'NN', 'VO', 'SVO', '1-GRAM'},
                                      gzipped=gzipped)
        except ValueError as e:
            logging.error('RED ALERT, RED ALERT')
            logging.error(e)
            continue
