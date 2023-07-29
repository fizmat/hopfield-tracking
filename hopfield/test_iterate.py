import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal_nulp, assert_allclose
from scipy.sparse import csr_matrix

from hopfield.iterate import annealing_curve, update_act_bulk, update_act_sequential


def test_annealing_curve():
    assert_array_almost_equal_nulp(annealing_curve(10, 40, 3, 2), [40., 20, 10, 10, 10])


class TestUpdateActBulk:
    @pytest.fixture
    def energy_matrix(self):
        return csr_matrix(np.array([[0, 1], [1, 0]]))

    @pytest.fixture
    def act(self):
        return np.array([0., 1.])

    def test_trivial(self, energy_matrix, act):
        update_act_bulk(energy_matrix, act)
        assert_allclose(act, [0.5 * (1 + np.tanh((- 2))), .5])

    def test_learning_rate(self, energy_matrix, act):
        update_act_bulk(energy_matrix, act, learning_rate=0.5)
        assert_allclose(act, [0.25 * (1 + np.tanh((- 2))), .75])

    def test_temperature(self, energy_matrix, act):
        update_act_bulk(energy_matrix, act, temperature=2.)
        assert_allclose(act, [0.5 * (1 + np.tanh((-1))), .5])

    def test_bias(self, energy_matrix, act):
        update_act_bulk(energy_matrix, act, bias=1.)
        assert_allclose(act, [0.5 * (1 + np.tanh((-1))), 0.5 * (1 + np.tanh(1))])


class TestUpdateActSequential:
    @pytest.fixture
    def energy_matrix(self):
        return csr_matrix(np.array([[0, 1], [1, 0]]))

    @pytest.fixture
    def act(self):
        return np.array([0., 1.])

    def test_trivial(self, energy_matrix, act):
        update_act_sequential(energy_matrix, act)
        a1 = 0.5 * (1 + np.tanh((- 2)))
        assert_allclose(act, [a1, 0.5 * (1 + np.tanh((-2 * a1)))])

    def test_temperature(self, energy_matrix, act):
        update_act_sequential(energy_matrix, act, temperature=2.)
        a1 = 0.5 * (1 + np.tanh((- 1)))
        assert_allclose(act, [a1, 0.5 * (1 + np.tanh((-a1)))])

    def test_bias(self, energy_matrix, act):
        update_act_sequential(energy_matrix, act, bias=1.)
        a1 = 0.5 * (1 + np.tanh((-1)))
        assert_allclose(act, [a1, 0.5 * (1 + np.tanh(-2 * a1 + 1))])
