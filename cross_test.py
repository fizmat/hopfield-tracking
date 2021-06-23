import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.sparse import csr_matrix

from cross import cross_energy, cross_energy_gradient, cross_energy_matrix, \
    segment_one_forks, segment_forks, segment_one_joins, segment_joins


def test_segment_one_forks():
    assert_array_equal(segment_one_forks(np.array([[0, 1]]), 0), np.empty((0, 2)))
    assert_array_equal(segment_one_forks(np.array([[0, 1], [0, 2]]), 0), [[0, 1]])
    assert_array_equal(segment_one_forks(np.array([[0, 2], [1, 2]]), 0), np.empty((0, 2)))
    segments = np.array([[0, 2], [0, 3], [1, 2], [1, 3]])
    assert_array_equal(segment_one_forks(segments, 2), [[2, 3]])


def test_segment_forks():
    null_segments = np.empty((0, 2), dtype=int)
    assert_array_equal(segment_forks(null_segments), np.empty((0, 2)))
    assert_array_equal(segment_forks(np.array([[0, 1]])), np.empty((0, 2)))
    assert_array_equal(segment_forks(np.array([[0, 1], [0, 2]])), [[0, 1], [1, 0]])
    assert_array_equal(segment_forks(np.array([[0, 2], [1, 2]])), np.empty((0, 2)))
    segments = np.array([[0, 2], [0, 3], [1, 2], [1, 3]])
    assert_array_equal(segment_forks(segments), [[0, 1], [1, 0], [2, 3], [3, 2]])


def test_segment_one_joins():
    assert_array_equal(segment_one_joins(np.array([[0, 1]]), 0), np.empty((0, 2)))
    assert_array_equal(segment_one_joins(np.array([[0, 1], [0, 2]]), 0), np.empty((0, 2)))
    assert_array_equal(segment_one_joins(np.array([[0, 2], [1, 2]]), 0), [[0, 1]])
    segments = np.array([[0, 2], [0, 3], [1, 2], [1, 3]])
    assert_array_equal(segment_one_joins(segments, 2), [[2, 0]])


def test_segment_joins():
    null_segments = np.empty((0, 2), dtype=int)
    assert_array_equal(segment_joins(null_segments), np.empty((0, 2)))
    assert_array_equal(segment_joins(np.array([[0, 1]])), np.empty((0, 2)))
    assert_array_equal(segment_joins(np.array([[0, 1], [0, 2]])), np.empty((0, 2)))
    assert_array_equal(segment_joins(np.array([[0, 2], [1, 2]])), [[0, 1], [1, 0]])
    segments = np.array([[0, 2], [0, 3], [1, 2], [1, 3]])
    assert_array_equal(segment_joins(segments), [[0, 2], [1, 3], [2, 0], [3, 1]])


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
