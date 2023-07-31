import numpy as np
from _pytest.python_api import approx
from numpy.testing import assert_array_equal, assert_array_almost_equal_nulp

from hopfield.energy.curvature import _curvature_pairwise, curvature_energy, curvature_energy_matrix, find_consecutive_segments
from hopfield.energy import energy, energy_gradient


def test_curvature_pairwise_empty():
    cosines, rab, rbc = _curvature_pairwise(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)))
    assert_array_equal(cosines, np.zeros(0))
    assert_array_equal(rab, np.zeros(0))
    assert_array_equal(rbc, np.zeros(0))


def test_curvature_pairwise():
    a = np.array([[0., 0], [0., 1], [0., 2], [0., 1]])
    b = np.array([[1., 0], [1., 1], [1., 2], [1., 1]])
    c = np.array([[2., 0], [2., 0], [3., 4], [0., 2]])
    r2 = np.sqrt(2)
    cosines, rab, rbc = _curvature_pairwise(a, b, c)
    assert_array_equal(cosines, [1, 1/r2, 1/r2, -1/r2])
    assert_array_equal(rab, [1, 1, 1, 1])
    assert_array_equal(rbc, [1, r2, 2*r2, r2])


def test_curvature_energy():
    assert_array_equal(curvature_energy(np.zeros((0,)), np.zeros((0,)), 1, 1), np.zeros(0))

    cosines = np.array([1, 1, 0.5, 0.5, -0.5])
    rr = np.array([1, 2, 1, 3, 5])
    assert_array_equal(curvature_energy(cosines, rr, 1., 1.),
                       [-1, -1/2, -1/8, -1/24, 0])
    assert_array_equal(curvature_energy(cosines, rr, 1., 1., cosine_power=1),
                       [-1, -1/2, -1/2, -1/6, 0])
    assert_array_almost_equal_nulp(curvature_energy(cosines, rr, 1., 1., cosine_power=0.5),
                                   [-1, -1/2, -1/np.sqrt(2), -np.sqrt(2)/6, 0])
    assert_array_equal(curvature_energy(cosines, rr, 1., 1., cosine_threshold=0.8),
                       [-1, -1/2, 0, 0, 0])
    assert_array_equal(curvature_energy(cosines, rr, 1., 1., distance_prod_power_in_denominator=0),
                       [-1, -1, -1/8, -1/8, 0])
    assert_array_equal(curvature_energy(cosines, rr, 3., 2., cosine_min_allowed=0.7),
                       [-2, -2/2, 3 - 2/8, 3 - 2/24, 3])


def test_find_adjacent_segments():
    null_segment = np.empty((0, 2), dtype=int)
    assert_array_equal(find_consecutive_segments(null_segment), np.empty((0, 2), dtype=int))
    segments = np.array([[1, 3], [1, 4], [2, 3]])
    assert_array_equal(find_consecutive_segments(segments), np.empty((0, 2), dtype=int))
    segments = np.array([[1, 3], [1, 4], [2, 3], [3, 4]])
    assert_array_equal(np.sort(find_consecutive_segments(segments), axis=0),
                       [[0, 3], [2, 3]])
    segments = np.array([[7, 11], [7, 12], [0, 1], [1, 2], [1, 3], [1, 4], [2, 5], [3, 6]])
    assert_array_equal(np.sort(find_consecutive_segments(segments), axis=0),
                       [[2, 3], [2, 4], [2, 5], [3, 6], [4, 7]])


def test_curvature_energy_matrix():
    null_pos = np.empty((0, 2))
    null_segment = np.empty((0, 2), dtype=int)
    w = curvature_energy_matrix(null_pos, null_segment, find_consecutive_segments(null_segment), 1., 1.)
    assert w.shape == (0, 0)

    pos = np.array([[0., 0], [1., 0], [2., 0], [2, 1], [2, -1], [3, 0], [3, 2]])
    w = curvature_energy_matrix(pos, null_segment, find_consecutive_segments(null_segment), 1., 1.)
    assert w.shape == (0, 0)

    seg = np.array([[7, 11], [7, 12], [0, 1], [1, 2], [1, 3], [1, 4], [2, 5], [3, 6]])
    w = curvature_energy_matrix(pos, seg, find_consecutive_segments(seg), 1., 1., do_sum_r=False)
    s = -1.  # straight cosine energy
    c = -1 / 4  # 45 degrees cosine energy
    l = -1 / 2  # straight diagonal
    assert_array_almost_equal_nulp(w.A, [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, s, c, c, 0, 0],
        [0, 0, s, 0, 0, 0, s, 0],
        [0, 0, c, 0, 0, 0, 0, l],
        [0, 0, c, 0, 0, 0, 0, 0],
        [0, 0, 0, s, 0, 0, 0, 0],
        [0, 0, 0, 0, l, 0, 0, 0],
    ], 8)


def test_curvature_energy_total():
    pos = np.array([[0., 0], [1., 0], [2., 0], [2, 1], [2, -1], [3, 0], [3, 2]])
    seg = np.array([[0, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6]])
    w = curvature_energy_matrix(pos, seg, find_consecutive_segments(seg), 1., 1., do_sum_r=False)
    assert energy(w, np.array([1, 1, 0, 0, 0, 0])) == -2
    assert energy(w, np.array([1, 0, 1, 0, 0, 0])) == approx(-1. / 2)
    assert energy(w, np.array([1, 0, 0, 1, 0, 0])) == approx(-1. / 2)
    assert energy(w, np.array([1, 0, 0, 0, 1, 0])) == approx(-1)
    assert energy(w, np.array([1, 0, 0, 0, 0, 1])) == approx(-1. / 4)
    assert energy(w, np.array([1, 1, 1, 1, 1, 1])) == approx(-17. / 4)
    assert energy(w, np.array([1, .1, .1, .1, .1, .1])) == approx(-1.7 / 4)


def test_curvature_energy_gradient():
    pos = np.array([[0., 0], [1., 0], [2., 0], [2, 1], [2, -1], [3, 0], [3, 2]])
    seg = np.array([[0, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6]])
    w = curvature_energy_matrix(pos, seg, find_consecutive_segments(seg), 1., 1., do_sum_r=False)
    act = np.array([1, 1, 0, 0, 0, 0])
    g = energy_gradient(w, act)
    assert_array_almost_equal_nulp(g, -np.array([2, 2, 1. / 2, 1. / 2, 1, 1. / 4]), 3)
