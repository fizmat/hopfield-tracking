import numpy as np
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix

from hopfield.energy.cross import cross_energy_matrix, segment_forks, segment_joins
from hopfield.energy import energy, energy_gradient


def test_segment_forks():
    null_segments = np.empty((0, 2), dtype=int)
    assert_array_equal(segment_forks(null_segments).todense(), np.empty((0, 0)))
    assert_array_equal(segment_forks(np.array([[0, 1]])).todense(), [[0]])
    assert_array_equal(segment_forks(np.array([[0, 1], [0, 2]])).todense(), [[0, 1],
                                                                             [1, 0]])
    assert_array_equal(segment_forks(np.array([[0, 2], [1, 2]])).todense(), np.zeros((2, 2)))
    segments = np.array([[0, 2], [0, 3], [1, 2], [1, 3]])
    assert_array_equal(segment_forks(segments).todense(), [[0, 1, 0, 0],
                                                           [1, 0, 0, 0],
                                                           [0, 0, 0, 1],
                                                           [0, 0, 1, 0]])


def test_segment_joins():
    null_segments = np.empty((0, 2), dtype=int)
    assert_array_equal(segment_joins(null_segments).todense(), np.empty((0, 0)))
    assert_array_equal(segment_joins(np.array([[0, 1]])).todense(), [[0]])
    assert_array_equal(segment_joins(np.array([[0, 1], [0, 2]])).todense(), np.zeros((2, 2)))
    assert_array_equal(segment_joins(np.array([[0, 2], [1, 2]])).todense(), [[0, 1],
                                                                             [1, 0]])
    segments = np.array([[0, 2], [0, 3], [1, 2], [1, 3]])
    assert_array_equal(segment_joins(segments).todense(), [[0, 0, 1, 0],
                                                           [0, 0, 0, 1],
                                                           [1, 0, 0, 0],
                                                           [0, 1, 0, 0]])


def test_cross_energy_matrix():
    null_segments = np.empty((0, 2), dtype=int)
    assert cross_energy_matrix(null_segments).shape == (0, 0)
    segments = np.array([[0, 1], [7, 11], [7, 12], [8, 13], [9, 13],
                         [5, 2], [5, 3], [6, 2], [6, 3]])
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
    assert energy(np.zeros((0, 0)), np.zeros(0)) == 0
    assert energy(np.zeros((1, 1)), np.array([0])) == 0
    assert energy(np.zeros((1, 1)), np.array([1])) == 0
    m = np.array([[0, 1], [1, 0]])
    assert energy(m, np.array([0, 0])) == 0
    assert energy(m, np.array([1, 0])) == 0
    assert energy(csr_matrix(m), np.array([0, 1])) == 0
    assert energy(m, np.array([1, 1])) == 2
    assert energy(csr_matrix(m), np.array([0.5, 1])) == 1


def test_cross_energy_gradient():
    m = np.array([[0, 1], [1, 0]])
    assert_array_equal(energy_gradient(m, np.array([0, 0])),
                       np.array([0, 0]))
    assert_array_equal(energy_gradient(m, np.array([1, 0])),
                       np.array([0, 2]))
    assert_array_equal(energy_gradient(csr_matrix(m), np.array([0, 1])),
                       np.array([2, 0]))
    assert_array_equal(energy_gradient(m, np.array([1, 1])),
                       np.array([2, 2]))
    assert_array_equal(energy_gradient(csr_matrix(m), np.array([0.5, 1])),
                       np.array([2, 1]))
