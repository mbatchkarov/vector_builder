from mock import Mock
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix, issparse
import pytest

from builder.composers.vectorstore import *
from discoutils.tokens import DocumentFeature, Token

DIM = 10

unigram_feature = DocumentFeature('1-GRAM', (Token('a', 'N'),))
unk_unigram_feature = DocumentFeature('1-GRAM', ((Token('unk', 'UNK')),))
bigram_feature = DocumentFeature('2-GRAM', (Token('a', 'N'), Token('b', 'V')))
unk_bigram_feature = DocumentFeature('2-GRAM', (Token('a', 'N'), Token('UNK', 'UNK')))
an_feature = DocumentFeature('AN', (Token('c', 'J'), Token('a', 'n')))
known_features = set([unigram_feature, bigram_feature, an_feature])
all_features = set([unigram_feature, bigram_feature, an_feature, unk_unigram_feature, unk_bigram_feature])


@pytest.fixture(scope='module')
def small_vectors():
    return Vectors.from_tsv('tests/resources/thesauri/small.txt.events.strings')


@pytest.fixture
def vectors_a():
    return Vectors.from_tsv('tests/resources/exp0-0a.strings')


@pytest.fixture
def left_comp(vectors_a):
    return LeftmostWordComposer(vectors_a)


@pytest.fixture
def right_comp(vectors_a):
    return RightmostWordComposer(vectors_a)


@pytest.fixture
def ones_vectors():
    return Vectors.from_tsv('tests/resources/ones.vectors.txt')


@pytest.fixture
def ones_vectors_no_pos():
    return Vectors.from_tsv('tests/resources/ones.vectors.nopos.txt',
                            enforce_word_entry_pos_format=False)


@pytest.fixture
def min_composer(ones_vectors):
    return MinComposer(ones_vectors)


@pytest.fixture
def max_composer(ones_vectors):
    return MaxComposer(ones_vectors)


def test_unigr_source_get_vector(small_vectors):
    # vectors come out right
    # a/N	amod:c	2   T:t1	1	T:t2	2	T:t3	3
    assert_array_equal(
        small_vectors.get_vector('a/N').todense(),
        [[0., 2., 0., 1., 2., 3., 0., 0.]]
    )

    # vocab is in sorted order
    assert ['also/RB', 'amod:c', 'now/RB', 't:t1', 't:t2', 't:t3', 't:t4', 't:t5', ] == small_vectors.columns

    assert small_vectors.get_vector('jfhjgjdfyjhgb/N') is None
    assert small_vectors.get_vector('jfhjgjdfyjhgb/J') is None


def test_unigr_source_contains(small_vectors):
    """
    Test if the unigram model only accepts unigram features
    """
    # for thing in (known_features | unk_unigram_feature | unk_bigram_feature):
    assert str(unigram_feature) in small_vectors
    for thing in (unk_unigram_feature, bigram_feature, unk_unigram_feature):
        assert str(thing) not in small_vectors


def test_additive_composer_contains(small_vectors):
    composer = AdditiveComposer(small_vectors)
    assert str(bigram_feature) in composer
    assert bigram_feature in composer

    for s in ['b/V_c/J', 'a/N_c/J', 'b/V_b/V_b/V']:
        assert s in composer
        assert str(s) in composer

    assert unigram_feature not in composer
    assert unk_unigram_feature not in composer
    assert unk_bigram_feature not in composer


def test_additive_composer_contains_nopos(ones_vectors_no_pos):
    for ch in 'abcd':
        assert ch in ones_vectors_no_pos

    composer = AdditiveComposer(ones_vectors_no_pos)
    for s in ['b_c', 'a_c', 'b_b_b', 'a_b']:
        assert composer.__contains__(s)  # todo s in composer

    assert 'd' not in composer
    assert 'd/N' not in composer
    assert 'a/N' not in composer


def test_add_mult_avg_compose_with_real_data(small_vectors):
    add = AdditiveComposer(small_vectors)
    mult = MultiplicativeComposer(small_vectors)
    avg = AverageComposer(small_vectors)

    assert_array_equal(
        np.array([[0, 0, 0, 0, 0, 9, 0, 0]]),
        mult.get_vector('a/N_b/V').A
    )

    assert_array_equal(
        np.array([[5, 2, 7, 1, 2, 6, 0, 0]]),
        add.get_vector('a/N_b/V').A
    )

    assert_array_equal(
        np.array([[5, 11, 15, 1, 2, 6, 10, 4]]),
        add.get_vector('a/N_b/V_c/J').A
    )

    assert_array_equal(
        np.array([[5, 2, 7, 1, 2, 6, 0, 0]]) / 2,
        avg.get_vector('a/N_b/V').A
    )


def test_add_mul_compose_with_synthetic_data():
    m = Mock()
    m.get_vector.return_value = csr_matrix(np.arange(DIM))

    add = AdditiveComposer(m)
    mult = MultiplicativeComposer(m)

    for i in range(1, 4):
        print(i)
        df = '_'.join(['a/N'] * i)
        result = add.get_vector(df)
        assert issparse(result)
        assert_array_equal(
            np.arange(DIM).reshape((1, DIM)) * i,
            result.A
        )

        result = mult.get_vector(df)
        assert issparse(result)
        assert_array_equal(
            np.arange(DIM).reshape((1, DIM)) ** i,
            result.A
        )


def test_min_max_composer(min_composer, max_composer):
    f1 = 'a/N_b/V_c/J'
    f2 = 'b/V_c/J'
    f3 = 'b/V'

    assert_array_equal(min_composer.get_vector(f1).A.ravel(),
                       np.array([0., 0., 0., 0.]))
    assert_array_equal(max_composer.get_vector(f1).A.ravel(),
                       np.array([1., 1., 1., 0.]))

    assert_array_equal(min_composer.get_vector(f2).A.ravel(),
                       np.array([0, 0, 0, 0]))
    assert_array_equal(max_composer.get_vector(f2).A.ravel(),
                       np.array([0, 1, 1, 0]))

    assert_array_equal(min_composer.get_vector(f3).A.ravel(),
                       np.array([0., 1., 0., 0.]))


def test_min_max_composer_contains(min_composer, max_composer):
    assert 'a/N_b/V_c/J' in max_composer
    assert 'b/V_c/J' in min_composer
    assert 'b/V_c/J_x/N' not in min_composer
    assert 'b/X_c/X' not in min_composer


def test_left_right_contains(left_comp, right_comp):
    for c in [left_comp, right_comp]:
        for f in ['like/V_fruit/N', 'fruit/N_cat/N', 'kid/N_like/V_fruit/N']:
            assert f in c

    assert 'cat/N' not in left_comp  # no unigrams
    assert 'cat/N' not in right_comp  # no unigrams
    assert 'red/J_cat/N' not in left_comp  # no unknown head words
    assert 'red/J_cat/N' in right_comp  # no unknown head words


def test_left_right_get_vector(left_comp, right_comp, vectors_a):
    v1 = left_comp.get_vector('like/V_fruit/N')
    v2 = right_comp.get_vector('like/V_fruit/N')

    assert v1.shape == v2.shape == (1, 7)
    assert_array_equal(v1.A.ravel(), np.array([0, 0, 0, 0, 0, 0, 0.11]))
    assert_array_equal(v2.A.ravel(), np.array([0.06, 0.05, 0, 0, 0, 0, 0]))
    assert_array_equal(v2.A, left_comp.get_vector('fruit/N').A)
    assert_array_equal(v2.A, vectors_a.get_vector('fruit/N').A)


def test_left_right_compose_all(left_comp):
    original_matrix, original_cols, original_rows = left_comp.unigram_source.to_sparse_matrix()
    matrix, cols, rows = left_comp.compose_all(['cat/N_game/N',
                                                DocumentFeature.from_string('dog/N_game/N'),
                                                'cat/N_a/N', 'cat/N_b/N', 'cat/N_c/N', 'cat/N_d/N', ])

    # the columns should remain unchanges
    assert original_cols == cols
    # the first rows are for the unigrams that existed before composition- 7 of them
    assert_array_equal(original_matrix.A, matrix.A[:7, :])
    # two new rows should appear, one for each composed feature
    # this should be reflected in both the index and the matrix
    assert rows['cat/N_game/N'] == 7
    assert rows['dog/N_game/N'] == 8
    assert matrix.shape == (13, 7) == (len(rows), len(cols))
    assert_array_equal(matrix.A[7, :], left_comp.unigram_source.get_vector('cat/N').A.ravel())
    assert_array_equal(matrix.A[8, :], left_comp.unigram_source.get_vector('dog/N').A.ravel())
    assert_array_equal(matrix.A[8, :], left_comp.unigram_source.get_vector('dog/N').A.ravel())

    for i in range(9, 12):
        assert_array_equal(matrix.A[i, :],
                           left_comp.unigram_source.get_vector('cat/N').A.ravel())


def test_verb_composer(ones_vectors):
    verb_composer = VerbComposer(ones_vectors)
    phrase = 'a/N_b/V_a/N'
    assert phrase in verb_composer
    assert_array_equal(verb_composer.get_vector(phrase).A.ravel(), np.array([0, 1, 0, 0]))
