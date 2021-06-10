import numpy as np
from _pytest.python_api import approx
from numpy.testing import assert_array_almost_equal, assert_array_equal

from curvature import curvature_energy_pairwise, curvature_layer_matrix, curvature_energy, curvature_energy_gradient


def test_curvature_energy_pairwise():
    a = np.array([[0., 0], [0., 1], [0., 2]])
    b = np.array([[1., 0], [1., 1], [1., 2]])
    c = np.array([[2., 0], [2, 0], [3, 4]])
    w = curvature_energy_pairwise(a, b, c)
    assert_array_almost_equal(w, [-1. / 2, -1. / 8, -1. / 16])


def test_curvature_energy_matrix():
    pos = np.array([[0., 0], [1., 0], [2., 0], [2, 1], [2, -1], [3, 0], [3, 2]])
    s1 = np.array([[0, 1]])
    s2 = np.array([[1, 2], [1, 3], [1, 4], [1, 5], [1, 6]])
    w = curvature_layer_matrix(pos, s1, s2)
    assert_array_equal(w.row, [0, 0, 0, 0, 0])
    assert_array_equal(w.col, [0, 1, 2, 3, 4])
    assert_array_almost_equal(w.data, [-1. / 2, -1. / 8, -1. / 8, -1. / 4, -1. / 16])


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