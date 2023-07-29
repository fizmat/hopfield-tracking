import numpy as np
from numpy.testing import assert_array_almost_equal_nulp
from pytest import approx

from hopfield.iterate import annealing_curve, update_layer_grad


def test_annealing_curve():
    assert_array_almost_equal_nulp(annealing_curve(10, 40, 3, 2), [40., 20, 10, 10, 10])


def test_update_layer_grad():
    grad = np.array([0., 1.])
    sigmoid_minus_one = 1 / (1 + np.exp(2.))
    assert sigmoid_minus_one == approx(0.5 * (1 + np.tanh(-1)))
    act = np.array([.5, .5])
    update_layer_grad(act, grad, 1.)
    assert_array_almost_equal_nulp(act, [.5, sigmoid_minus_one], 2)
    act = np.array([.5, .5])
    update_layer_grad(act, grad, 1., learning_rate=0.5)
    assert_array_almost_equal_nulp(act, [.5, (0.5 + sigmoid_minus_one) / 2], 2)
    act = np.array([.5, .5])
