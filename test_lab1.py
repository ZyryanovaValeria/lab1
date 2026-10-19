git addimport numpy as np
from grader_contracts.python_basics import PositiveIntegerInput as P, TextInput as T, VectorPairInput as V
from grader_contracts.numpy_tasks import *
from python_basics import *
import numpy_tasks as nt


def test_count_vowels():
    assert count_vowels(T("")) == 0
    assert count_vowels(T("AEIOU")) == 5
    assert count_vowels(T("rhythm")) == 0
    assert count_vowels(T("Hello World")) == 3


def test_unique_chars():
    assert has_unique_characters(T("")) is True
    assert has_unique_characters(T("abc")) is True
    assert has_unique_characters(T("aba")) is False
    assert has_unique_characters(T("aA")) is True


def test_one_bits():
    assert count_one_bits(P(1)) == 1
    assert count_one_bits(P(7)) == 3
    assert count_one_bits(P(1024)) == 1
    assert count_one_bits(P(255)) == 8


def test_persistence():
    assert multiplicative_persistence(P(39)) == 3
    assert multiplicative_persistence(P(4)) == 0
    assert multiplicative_persistence(P(999)) == 4
    assert multiplicative_persistence(P(10)) == 1


def test_mse():
    assert mse(V([1, 2, 3], [1, 2, 3])) == 0
    assert mse(V([1, 2], [3, 4])) == 4
    assert mse(V([0], [2])) == 4


def test_prime_factorization():
    assert prime_factorization(P(86240)) == "(2**5)(5)(7**2)(11)"
    assert prime_factorization(P(2)) == "(2)"
    assert prime_factorization(P(97)) == "(97)"
    assert prime_factorization(P(1024)) == "(2**10)"
    assert prime_factorization(P(12)) == "(2**2)(3)"


def test_pyramid():
    assert pyramid(P(1)) == 1
    assert pyramid(P(5)) == 2
    assert pyramid(P(14)) == 3
    assert pyramid(P(55)) == 5
    assert pyramid(P(2)) == "It is impossible"
    assert pyramid(P(100)) == "It is impossible"


def test_balanced():
    assert is_balanced_number(P(1234006)) is True
    assert is_balanced_number(P(123456)) is False
    assert is_balanced_number(P(5)) is True
    assert is_balanced_number(P(12)) is True
    assert is_balanced_number(P(1221)) is True
    assert is_balanced_number(P(1232)) is False


def test_sum_prod():
    A = np.array([np.eye(2), 2 * np.eye(2)])
    x = np.array([[[1], [2]], [[3], [4]]])
    r = nt.sum_prod(MatrixVectorBatchInput(A, x))
    assert r.shape == (2, 1)
    assert (r == [[7], [10]]).all()
    r1 = nt.sum_prod(MatrixVectorBatchInput(np.array([[[3]]]), np.array([[[2]]])))
    assert r1.tolist() == [[6]]


def test_binarize():
    M = np.array([[0.1, 0.5], [0.6, 2]])
    r = nt.binarize(BinarizeInput(M))
    assert r.tolist() == [[0, 0], [1, 1]]
    assert M[0, 0] == 0.1


def test_unique():
    M = np.array([[1, 1, 2], [3, 3, 3]])
    assert nt.unique_rows(MatrixInput(M)) == [[1, 2], [3]]
    assert nt.unique_columns(MatrixInput(M)) == [[1, 3], [1, 3], [2, 3]]
    assert nt.unique_rows(MatrixInput(np.array([[7]]))) == [[7]]


def test_matrix_statistics():
    s = nt.matrix_statistics(RandomMatrixInput(3, 4, 5.0, 2.0, 1))
    assert s.matrix.shape == (3, 4)
    assert s.row_means.shape == (3,) and s.column_means.shape == (4,)
    assert s.row_variances.shape == (3,) and s.column_variances.shape == (4,)
    s2 = nt.matrix_statistics(RandomMatrixInput(3, 4, 5.0, 2.0, 1))
    assert (s.matrix == s2.matrix).all()
    assert np.allclose(s.row_means, s.matrix.mean(axis=1))


def test_chess():
    assert nt.chess(ChessInput(3, 4, 0, 1)).tolist() == [[0,1,0,1],[1,0,1,0],[0,1,0,1]]
    assert nt.chess(ChessInput(1, 1, 5, 9)).tolist() == [[5]]


def test_rectangle():
    img = nt.draw_rectangle(RectangleInput(2, 2, 4, 4, (255, 0, 0), (0, 0, 0)))
    assert img.shape == (4, 4, 3)
    assert (img[1:3, 1:3] == [255, 0, 0]).all()
    assert (img[0] == 0).all()


def test_ellipse():
    img = nt.draw_ellipse(EllipseInput(2, 1, 5, 7, (1, 2, 3), (0, 0, 0)))
    assert img.shape == (5, 7, 3)
    assert tuple(img[2, 3]) == (1, 2, 3)
    assert tuple(img[0, 0]) == (0, 0, 0)


def test_time_series():
    r = nt.analyze_time_series(TimeSeriesInput([1, 3, 2, 4, 1], 2))
    assert r.mean == 2.2
    assert r.local_maxima_indices == [1, 3]
    assert r.local_minima_indices == [2]
    assert np.allclose(r.moving_average, [2, 2.5, 3, 2.5])
    assert np.isclose(r.std, np.sqrt(r.variance))
    flat = nt.analyze_time_series(TimeSeriesInput([1, 1, 1], 3))
    assert flat.local_maxima_indices == [] and flat.local_minima_indices == []
    assert len(flat.moving_average) == 1


def test_one_hot():
    assert nt.one_hot(OneHotInput(np.array([0, 2, 3, 0]))).tolist() == [[1,0,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]]
    assert nt.one_hot(OneHotInput(np.array([0, 1]), 4)).shape == (2, 4)
