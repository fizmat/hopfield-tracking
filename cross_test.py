import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.sparse import csr_matrix

from cross import fork_layer_matrix, join_layer_matrix, cross_energy, cross_energy_gradient, cross_energy_matrix


def test_fork_layer_matrix():
    m = fork_layer_matrix(np.array([]))
    assert_array_equal(m.todense(), np.zeros((0, 0)))
    m = fork_layer_matrix(np.array([[0, 1]]))
    assert_array_equal(m.todense(), [[0]])
    m = fork_layer_matrix(np.array([[0, 1], [0, 2]]))
    assert_array_equal(m.todense(), [[0, 1], [1, 0]])
    m = fork_layer_matrix(np.array([[0, 2], [1, 2]]))
    assert_array_equal(m.todense(), [[0, 0], [0, 0]])

    segments = np.array([[0, 2], [0, 3], [1, 2], [1, 3]])
    m = fork_layer_matrix(segments)
    assert_array_equal(m.todense(), [[0, 1, 0, 0],
                                     [1, 0, 0, 0],
                                     [0, 0, 0, 1],
                                     [0, 0, 1, 0]])


def test_join_layer_matrix():
    m = join_layer_matrix(np.array([]))
    assert_array_equal(m.todense(), np.zeros((0, 0)))
    m = join_layer_matrix(np.array([[0, 1]]))
    assert_array_equal(m.todense(), [[0]])
    m = join_layer_matrix(np.array([[0, 1], [0, 2]]))
    assert_array_equal(m.todense(), [[0, 0], [0, 0]])
    m = join_layer_matrix(np.array([[0, 2], [1, 2]]))
    assert_array_equal(m.todense(), [[0, 1], [1, 0]])

    segments = np.array([[0, 2], [0, 3], [1, 2], [1, 3]])
    m = join_layer_matrix(segments)
    assert_array_equal(m.todense(), [[0, 0, 1, 0],
                                     [0, 0, 0, 1],
                                     [1, 0, 0, 0],
                                     [0, 1, 0, 0]])


def test_cross_energy_matrix():
    assert cross_energy_matrix([]).shape == (0, 0)
    assert cross_energy_matrix([np.array([]), np.array([])]).shape == (0, 0)
    segments = [np.array([]), np.array([[0, 1]]),
                np.array([[7, 11], [7, 12]]), np.array([[8, 13], [9, 13]]),
                np.array([[5, 2], [5, 3], [6, 2], [6, 3]])]
    assert_array_equal(cross_energy_matrix(segments).A,
                       np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 1, 1, 0],
                                 [0, 0, 0, 0, 0, 1, 0, 0, 1],
                                 [0, 0, 0, 0, 0, 1, 0, 0, 1],
                                 [0, 0, 0, 0, 0, 0, 1, 1, 0]]))


def test_cross_energy():
    assert cross_energy(np.zeros((0, 0)), np.zeros(0)) == 0
    assert cross_energy(np.zeros((1, 1)), np.array([0])) == 0
    assert cross_energy(np.zeros((1, 1)), np.array([1])) == 0
    m = np.array([[0, 1], [1, 0]])
    assert cross_energy(m, np.array([0, 0])) == 0
    assert cross_energy(m, np.array([1, 0])) == 0
    assert cross_energy(csr_matrix(m), np.array([0, 1])) == 0
    assert cross_energy(m, np.array([1, 1])) == 1
    assert cross_energy(csr_matrix(m), np.array([0.5, 1])) == 0.5


def test_cross_energy_gradient():
    m = np.array([[0, 1], [1, 0]])
    assert_array_almost_equal(cross_energy_gradient(m, np.array([0, 0])),
                              np.array([0, 0]))
    assert_array_almost_equal(cross_energy_gradient(m, np.array([1, 0])),
                              np.array([0, 1]))
    assert_array_almost_equal(cross_energy_gradient(csr_matrix(m), np.array([0, 1])),
                              np.array([1, 0]))
    assert_array_almost_equal(cross_energy_gradient(m, np.array([1, 1])),
                              np.array([1, 1]))
    assert_array_almost_equal(cross_energy_gradient(csr_matrix(m), np.array([0.5, 1])),
                              np.array([1, 0.5]))
