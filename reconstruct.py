from typing import List, Union

import numpy as np
from numpy import ndarray
from scipy.sparse import spmatrix


def annealing_curve(t_min, t_max, cooling_steps, rest_steps):
    return np.concatenate([
        np.geomspace(t_max, t_min, cooling_steps),
        np.full(rest_steps, t_min)])


def update_layer_grad(act: ndarray, grad: ndarray, t: float, dropout_rate: float = 0.,
                      learning_rate: float = 1., bias: float = 0.) -> None:
    n = len(act)
    if dropout_rate:
        not_dropout = np.random.choice(n, round(n * (1. - dropout_rate)), replace=False)
        next_act = 0.5 * (1 + np.tanh((- grad[not_dropout] + bias) / t))
        updated_act = next_act * learning_rate + act[not_dropout] * (1. - learning_rate)
        act[not_dropout] = updated_act
    else:
        next_act = 0.5 * (1 + np.tanh((- grad + bias) / t))
        act[:] = next_act * learning_rate + act * (1. - learning_rate)


def should_stop(act: ndarray, acts: List[ndarray], min_act_change: float = 1e-5, lookback: int = 1) -> bool:
    return max(np.max(act - a0) for a0 in acts[-lookback:]) < min_act_change


def energy(matrix: Union[spmatrix, ndarray], act: ndarray):
    return matrix.dot(act).dot(act)


def energy_gradient(matrix: Union[spmatrix, ndarray], act: ndarray):
    return 2 * matrix.dot(act)
