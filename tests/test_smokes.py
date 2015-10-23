from builder.generate_random_vectors import generate
from discoutils.thesaurus_loader import Vectors


def test_random_vectors(tmpdir):
    output = str(tmpdir.join('vectors.h5'))
    generate(output, 10)
    v = Vectors.from_tsv(output)
    assert v.matrix.shape[1] == 10


