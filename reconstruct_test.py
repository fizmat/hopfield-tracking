import numpy as np
import torch_testing as tt
from numpy.testing import assert_array_almost_equal_nulp, assert_array_almost_equal
from pytest import approx
from torch import tensor

from reconstruct import annealing_curve, flatten_act, mean_act, dist_act, should_stop, update_layer


def test_annealing_curve():
    assert_array_almost_equal_nulp(annealing_curve(10, 40, 3, 2), [40., 20, 10, 10, 10])


def test_update_layer():
    act = tensor([.5, .5], requires_grad=True)
    e = - 2 * act.sum()
    e.backward()
    result = update_layer(act, 1.).detach().numpy()
    # not obvious at all
    assert_array_almost_equal(result, 1 / (1 + np.exp(np.array([-4., -4.]))))


def test_flatten_act():
    act = [tensor([1, 2, 3]), tensor([4, 5, 6])]
    tt.assert_equal(flatten_act(act), tensor([1, 2, 3, 4, 5, 6]))


def test_mean_act():
    act = [tensor([1., 3]), tensor([5., 4, 5, 6])]
    assert mean_act(act) == 4


def test_dist_act():
    act = [tensor([1., 2., 3.])]
    perf = [tensor([1., 3., 5.])]
    assert dist_act(act, perf) == approx(5. / 3.)


def test_should_stop():
    act1 = [tensor([1., 2, 3]), tensor([4., 5, 6])]
    act2 = [tensor([1., 2, 3.01]), tensor([4., 5, 6])]
    assert should_stop(act1, act1)
    assert should_stop(act2, act2)
    assert not should_stop(act1, act2)
    assert not should_stop(act2, act1)
    assert should_stop(act1, act2, 0.1)
    assert should_stop(act2, act1, 0.1)
