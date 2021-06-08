import numpy as np
from numpy.testing import assert_array_almost_equal
from pytest import approx

from reconstruct import annealing_curve, flatten_act, mean_act, dist_act, should_stop, update_layer_grad


def test_annealing_curve():
    assert_array_almost_equal(annealing_curve(10, 40, 3, 2), [40., 20, 10, 10, 10])


def test_update_layer_grad():
    act = np.array([.5, .5])
    grad = np.array([0., 1.])
    sigmoid_minus_one = 1 / (1 + np.exp(2.))
    assert sigmoid_minus_one == approx(0.5 * (1 + np.tanh(-1)))
    assert_array_almost_equal(update_layer_grad(act, grad, 1.), [.5, sigmoid_minus_one])
    assert_array_almost_equal(update_layer_grad(act, grad, 1., learning_rate=0.5), [.5, (0.5 + sigmoid_minus_one) / 2])
    assert_array_almost_equal(update_layer_grad(act, grad, 1., dropout_rate=1.), [.5, .5])


def test_flatten_act():
    act = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    assert_array_almost_equal(flatten_act(act), [1, 2, 3, 4, 5, 6])


def test_mean_act():
    act = [np.array([1., 3]), np.array([5., 4, 5, 6])]
    assert mean_act(act) == 4


def test_dist_act():
    act = [np.array([1., 2., 3.])]
    perf = [np.array([1., 3., 5.])]
    assert dist_act(act, perf) == approx(5. / 3.)


def test_should_stop():
    act1 = [np.array([1., 2, 3]), np.array([4., 5, 6])]
    act2 = [np.array([1., 2, 3.01]), np.array([4., 5, 6])]
    assert should_stop(act1, act1)
    assert should_stop(act2, act2)
    assert not should_stop(act1, act2)
    assert not should_stop(act2, act1)
    assert should_stop(act1, act2, 0.1)
    assert should_stop(act2, act1, 0.1)
