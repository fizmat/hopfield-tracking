import numpy as np
import pytest
from _pytest.python_api import approx
from numpy.testing import assert_array_equal, assert_allclose

from hopfield.energy import energy, energy_gradient
from hopfield.energy.curvature import _curvature_pairwise, _curvature_energy_pairwise, curvature_energy_matrix, \
    find_consecutive_segments, _kink_energy_pairwise, prep_curvature


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
    assert_array_equal(cosines, [1, 1 / r2, 1 / r2, -1 / r2])
    assert_array_equal(rab, [1, 1, 1, 1])
    assert_array_equal(rbc, [1, r2, 2 * r2, r2])


def test_curvature_energy_pairwise_empty():
    assert_array_equal(_curvature_energy_pairwise(np.zeros((0,)), np.zeros((0,)), np.zeros((0,)), 1., 0., 0., True),
                       np.zeros(0))
    assert_array_equal(_curvature_energy_pairwise(np.zeros((0,)), np.zeros((0,)), np.zeros((0,)), 1., 0., 0., False),
                       np.zeros(0))


class TestCurvatureEnergyPairwise:
    @pytest.fixture
    def cosines(self):
        return np.array([1, 1, 0.5, 0.5, -0.5])

    @pytest.fixture
    def r1(self):
        return np.array([1, 2, 1, 3, 5])

    @pytest.fixture
    def r2(self):
        return np.zeros(5)

    def test_default(self, cosines, r1, r2):
        assert_array_equal(_curvature_energy_pairwise(cosines, r1, r2),
                           [1, 1 / 2, 1 / 8, 1 / 24, 0])

    def test_cosine_power(self, cosines, r1, r2):
        assert_array_equal(_curvature_energy_pairwise(cosines, r1, r2, cosine_power=1.),
                           [1, 1 / 2, 1 / 2, 1 / 6, 0])

    def test_cosine_threshold(self, cosines, r1, r2):
        assert_array_equal(_curvature_energy_pairwise(cosines, r1, r2, cosine_threshold=0.8),
                           [1, 1 / 2, 0, 0, 0])

    def test_distance_power(self, cosines, r1, r2):
        assert_array_equal(_curvature_energy_pairwise(cosines, r1, r2, distance_power=0),
                           [1, 1, 1 / 8, 1 / 8, 0])

    def test_distance_sum(self, cosines, r1):
        assert_array_equal(_curvature_energy_pairwise(cosines, r1, np.ones(5)),
                           [1 / 2, 1 / 3, 1 / 16, 1 / 32, 0])

    def test_distance_mult(self, cosines, r1):
        assert_array_equal(_curvature_energy_pairwise(cosines, r1, np.ones(5), sum_distances=False),
                           [1, 1 / 2, 1 / 8, 1 / 24, 0])

    def test_kink_default(self, cosines):
        assert_array_equal(_kink_energy_pairwise(cosines),
                           [0, 0, 0, 0, 1])

    def test_kink_none(self, cosines):
        assert_array_equal(_kink_energy_pairwise(cosines, -1.),
                           [0, 0, 0, 0, 0])

    def test_kink_aggressive(self, cosines):
        assert_array_equal(_kink_energy_pairwise(cosines, 0.6),
                           [0, 0, 1, 1, 1])


class TestFindConsecutiveSegments:
    def test_empty(self):
        null_segment = np.empty((0, 2), dtype=int)
        assert_array_equal(find_consecutive_segments(null_segment), np.empty((0, 2), dtype=int))

    def test_not_found(self):
        segments = np.array([[1, 3], [1, 4], [2, 3]])
        assert_array_equal(find_consecutive_segments(segments), np.empty((0, 2), dtype=int))

    def test_small(self):
        segments = np.array([[1, 3], [1, 4], [2, 3], [3, 4]])
        assert_array_equal(np.sort(find_consecutive_segments(segments), axis=0),
                           [[0, 3], [2, 3]])

    def test_nonconsecutive_hit_ids(self):
        segments = np.array([[7, 11], [7, 12], [0, 1], [1, 2], [1, 3], [1, 4], [2, 5], [3, 6]])
        assert_array_equal(np.sort(find_consecutive_segments(segments), axis=0),
                           [[2, 3], [2, 4], [2, 5], [3, 6], [4, 7]])


class TestCurvatureEnergyMatrix:
    def test_zero_size(self):
        w = curvature_energy_matrix(0, np.empty((0, 2)), np.empty(0),
                                    np.empty(0), np.empty(0))
        assert w.shape == (0, 0)

    def test_empty(self):
        w = curvature_energy_matrix(5, np.empty((0, 2)), np.empty(0),
                                    np.empty(0), np.empty(0))
        assert_array_equal(w.toarray(), np.zeros((5, 5)))

    def test_actual_matrix(self):
        pairs = np.array([[0, 1], [1, 2], [1, 3]])
        cosines = np.array([0, 0.5, 1.])
        r1 = np.array([1, 1, 3])
        r2 = np.array([1, 2, 1])
        w = curvature_energy_matrix(4, pairs, cosines, r1, r2)
        assert_allclose(w.toarray(), [[0, 0, 0, 0],
                                      [0, 0, 1 / 24, 1 / 4],
                                      [0, 1 / 24, 0, 0],
                                      [0, 1 / 4, 0, 0]])

    def test_big_matrix(self):
        seg = np.array([[7, 11], [7, 12], [0, 1], [1, 2], [1, 3], [1, 4], [2, 5], [3, 6]])
        pos = np.array([[0., 0], [1., 0], [2., 0], [2, 1], [2, -1], [3, 0], [3, 2]])
        pairs, cosines, r1, r2 = prep_curvature(pos, seg)
        w = curvature_energy_matrix(len(seg), pairs, cosines, r1, r2, do_sum_distances=False)
        s = 1.  # straight cosine energy
        c = 1 / 4  # 45 degrees cosine energy
        d = 1 / 2  # straight diagonal
        assert_allclose(w.toarray(), [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, s, c, c, 0, 0],
            [0, 0, s, 0, 0, 0, s, 0],
            [0, 0, c, 0, 0, 0, 0, d],
            [0, 0, c, 0, 0, 0, 0, 0],
            [0, 0, 0, s, 0, 0, 0, 0],
            [0, 0, 0, 0, d, 0, 0, 0],
        ])


class TestCurvatureEnergyResult:
    @pytest.fixture
    def pos(self):
        return np.array([[0., 0], [1., 0], [2., 0], [2, 1], [2, -1], [3, 0], [3, 2]])

    @pytest.fixture
    def seg(self):
        return np.array([[0, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6]])

    def test_curvature_energy_total(self, pos, seg):
        pairs, cosines, r1, r2, = prep_curvature(pos, seg)
        w = curvature_energy_matrix(len(seg), pairs, cosines, r1, r2, do_sum_distances=False)
        assert energy(w, np.array([1, 1, 0, 0, 0, 0])) == 2
        assert energy(w, np.array([1, 0, 1, 0, 0, 0])) == approx(1. / 2)
        assert energy(w, np.array([1, 0, 0, 1, 0, 0])) == approx(1. / 2)
        assert energy(w, np.array([1, 0, 0, 0, 1, 0])) == approx(1)
        assert energy(w, np.array([1, 0, 0, 0, 0, 1])) == approx(1. / 4)
        assert energy(w, np.array([1, 1, 1, 1, 1, 1])) == approx(17. / 4)
        assert energy(w, np.array([1, .1, .1, .1, .1, .1])) == approx(1.7 / 4)

    def test_curvature_energy_gradient(self, pos, seg):
        pairs, cosines, r1, r2, = prep_curvature(pos, seg)
        w = curvature_energy_matrix(len(seg), pairs, cosines, r1, r2,  do_sum_distances=False)
        act = np.array([1, 1, 0, 0, 0, 0])
        g = energy_gradient(w, act)
        assert_allclose(g, np.array([2, 2, 1. / 2, 1. / 2, 1, 1. / 4]))
