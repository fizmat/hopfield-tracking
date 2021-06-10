import numpy as np
from _pytest.python_api import approx
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_array_almost_equal_nulp

from curvature import curvature_energy_pairwise, curvature_layer_matrix, curvature_energy, curvature_energy_gradient, \
    curvature_energy_matrix


def test_curvature_energy_pairwise():
    assert_array_equal(curvature_energy_pairwise(np.zeros(0), np.zeros(0), np.zeros(0)),
                       np.zeros(0))

    a = np.array([[0., 0], [0., 1], [0., 2]])
    b = np.array([[1., 0], [1., 1], [1., 2]])
    c = np.array([[2., 0], [2, 0], [3, 4]])
    r2 = np.sqrt(2)
    assert_array_almost_equal(curvature_energy_pairwise(a, b, c),
                              [-0.5, -0.5 * (0.5 / r2) / r2, -0.5 * (0.5 / r2) / 2 / r2])
    assert_array_almost_equal(curvature_energy_pairwise(a, b, c, power=1),
                              [-0.5, -0.5 * (1 / r2) / r2, -0.5 * (1 / r2) / 2 / r2])
    assert_array_almost_equal(curvature_energy_pairwise(a, b, c, cosine_threshold=0.8),
                              [-0.5, 0, 0])


def test_curvature_layer_matrix():
    w = curvature_layer_matrix(np.array([]), np.array([]), np.array([]))
    assert w.shape == (0, 0)
    pos = np.array([[0., 0], [1., 0], [2., 0], [2, 1], [2, -1], [3, 0], [3, 2]])
    s1 = np.array([[0, 1]])
    s2 = np.array([[1, 2], [1, 3], [1, 4], [1, 5], [1, 6]])
    w = curvature_layer_matrix(pos, np.array([]), np.array([]))
    assert w.shape == (0, 0)
    w = curvature_layer_matrix(pos, s1, np.array([]))
    assert w.shape == (1, 0)
    w = curvature_layer_matrix(pos, np.array([]), s2)
    assert w.shape == (0, 5)
    w = curvature_layer_matrix(pos, s1, s2)
    assert_array_almost_equal_nulp(w.A, [[-1. / 2, -1. / 8, -1. / 8, -1. / 4, -1. / 16]], 4)
    s3 = np.array([[1, 2], [1, 3], [2, 4], [2, 5], [2, 6]])
    w = curvature_layer_matrix(pos, s1, s3)
    assert_array_almost_equal_nulp(w.A, [[-1. / 2, -1. / 8, 0, 0, 0]], 4)
    assert w.getnnz() == 2


def test_curvature_energy_matrix():
    w = curvature_energy_matrix(np.array([]), [])
    assert w.shape == (0, 0)

    pos = np.array([[0., 0], [1., 0], [2., 0], [2, 1], [2, -1], [3, 0], [3, 2]])
    w = curvature_energy_matrix(pos, [])
    assert w.shape == (0, 0)

    segments = [np.array([[7, 11], [7, 12]]), np.array([]), np.array([[0, 1]]),
                np.array([[1, 2], [1, 3], [1, 4]]),
                np.array([[2, 5], [3, 6]])]
    w = curvature_energy_matrix(pos, segments)
    s = -1 / 2  # straight cosine energy
    c = -1 / 8  # 45 degrees cosine energy
    l = - 1 / 4  # straight diagonal
    assert_array_almost_equal_nulp(w.A, [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, s, c, c, 0, 0],
        [0, 0, 0, 0, 0, 0, s, 0],
        [0, 0, 0, 0, 0, 0, 0, l],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], 10)


def test_curvature_energy():
    pos = np.array([[0., 0], [1., 0], [2., 0], [2, 1], [2, -1], [3, 0], [3, 2]])
    s1 = np.array([[0, 1]])
    s2 = np.array([[1, 2], [1, 3], [1, 4], [1, 5], [1, 6]])
    w = curvature_layer_matrix(pos, s1, s2)
    first = np.array([1])
    assert curvature_energy(w, first, np.array([1, 0, 0, 0, 0])) == - 0.5
    assert curvature_energy(w, first, np.array([0, 1, 0, 0, 0])) == approx(- 1. / 8)
    assert curvature_energy(w, first, np.array([0, 0, 1, 0, 0])) == approx(- 1. / 8)
    assert curvature_energy(w, first, np.array([0, 0, 0, 1, 0])) == approx(- 1. / 4)
    assert curvature_energy(w, first, np.array([0, 0, 0, 0, 1])) == approx(- 1. / 16)
    assert curvature_energy(w, first, np.array([1, 1, 1, 1, 1])) == approx(- 17. / 16)
    assert curvature_energy(w, first, np.array([.1, .1, .1, .1, .1])) == approx(- 1.7 / 16)


def test_curvature_energy_gradient():
    pos = np.array([[0., 0], [1., 0], [2., 0], [2, 1], [2, -1], [3, 0], [3, 2]])
    s1 = np.array([[0, 1]])
    s2 = np.array([[1, 2], [1, 3], [1, 4], [1, 5], [1, 6]])
    w = curvature_layer_matrix(pos, s1, s2)
    first = np.array([1.])
    second = np.array([1., 0, 0, 0, 0])
    g1, g2 = curvature_energy_gradient(w, first, second)
    assert_array_almost_equal(g1, np.array([-0.5]))
    assert_array_almost_equal(g2, np.array([- 0.5, -1. / 8, -1. / 8, -1. / 4, -1. / 16]))
