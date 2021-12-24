from typing import List

import numpy as np
# from memory_profiler import profile
from numpy import ndarray

from hopfield.energy import energy_gradient
from hopfield.energy.cross import cross_energy_matrix
from hopfield.energy.curvature import segment_adjacent_pairs, curvature_energy_matrix


def construct_energy_matrix(config, pos, seg):
    pairs = segment_adjacent_pairs(seg)
    crossing_matrix = cross_energy_matrix(seg, pos, config['cosine_min_allowed'], pairs)
    curvature_matrix = curvature_energy_matrix(pos, seg, pairs,
                                               config['cosine_power'], config['cosine_min_rewarded'],
                                               config['distance_power'])
    crossing_part = config['alpha'] / 2 * crossing_matrix
    curvature_part = config['gamma'] / 2 * curvature_matrix
    return crossing_part - curvature_part, crossing_part, curvature_part


def construct_anneal(config):
    return annealing_curve(config['tmin'], config['tmax'], config['anneal_steps'],
                           config['total_steps'] - config['anneal_steps'])


# @profile
def hopfield_iterate(config, energy_matrix, temp_curve, seg):
    act = np.full(len(seg), config['starting_act'])
    for i, t in enumerate(temp_curve):
        grad = energy_gradient(energy_matrix, act)
        update_layer_grad(act, grad, t, config['dropout'], config['learning_rate'], config['bias'])
    return act


def hopfield_history(config, energy_matrix, temp_curve, seg):
    act = np.full(len(seg), config['starting_act'])
    acts = [act.copy()]
    for i, t in enumerate(temp_curve):
        grad = energy_gradient(energy_matrix, act)
        update_layer_grad(act, grad, t, config['dropout'], config['learning_rate'], config['bias'])
        acts.append(act.copy())
    return acts


def annealing_curve(t_min: float, t_max: float, cooling_steps: int, rest_steps: int) -> np.ndarray:
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
