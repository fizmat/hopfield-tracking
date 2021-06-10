import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.sparse import csr_matrix

from cross import fork_energy_matrix, join_energy_matrix, layer_energy, layer_energy_gradient


def test_fork_matrix():
    segments = np.array([[0, 2], [0, 3], [1, 2], [1, 3]])
    m = fork_energy_matrix(segments)
    assert_array_equal(m.todense(), [[0, 1, 0, 0],
                                     [1, 0, 0, 0],
                                     [0, 0, 0, 1],
                                     [0, 0, 1, 0]])


def test_join_matrix():
    segments = np.array([[0, 2], [0, 3], [1, 2], [1, 3]])
    m = join_energy_matrix(segments)
    assert_array_equal(m.todense(), [[0, 0, 1, 0],
                                     [0, 0, 0, 1],
                                     [1, 0, 0, 0],
                                     [0, 1, 0, 0]])


def test_layer_energy():
    m = np.array([[0, 1], [1, 0]])
    assert layer_energy(m, np.array([0, 0])) == 0
    assert layer_energy(m, np.array([1, 0])) == 0
    assert layer_energy(csr_matrix(m), np.array([0, 1])) == 0
    assert layer_energy(m, np.array([1, 1])) == 1
    assert layer_energy(csr_matrix(m), np.array([0.5, 1])) == 0.5


def test_layer_energy_gradient():
    m = np.array([[0, 1], [1, 0]])
    assert_array_almost_equal(layer_energy_gradient(m, np.array([0, 0])),
                              np.array([0, 0]))
    assert_array_almost_equal(layer_energy_gradient(m, np.array([1, 0])),
                              np.array([0, 1]))
    assert_array_almost_equal(layer_energy_gradient(csr_matrix(m), np.array([0, 1])),
                              np.array([1, 0]))
    assert_array_almost_equal(layer_energy_gradient(m, np.array([1, 1])),
                              np.array([1, 1]))
    assert_array_almost_equal(layer_energy_gradient(csr_matrix(m), np.array([0.5, 1])),
                              np.array([1, 0.5]))
