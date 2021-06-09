from typing import List

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy.stats import bernoulli


def annealing_curve(t_min, t_max, cooling_steps, rest_steps):
    return np.concatenate([
        np.geomspace(t_max, t_min, cooling_steps),
        np.full(rest_steps, t_min)])


def update_layer_grad(act: ndarray, grad: ndarray, t: float, dropout_rate: float = 0.,
                      learning_rate: float = 1., bias: float = 0.) -> ndarray:
    next_act = 0.5 * (1 + np.tanh((- grad + bias)/ t))
    updated_act = next_act * learning_rate + act * (1. - learning_rate)
    dropout = bernoulli.rvs(dropout_rate)
    return np.where(dropout, act, updated_act)


def flatten_act(act: List[ndarray]) -> ndarray:
    return np.concatenate([a.flatten() for a in act])


def mean_act(act: List[ndarray]) -> float:
    return flatten_act(act).mean().item()


def dist_act(act1: List[ndarray], act2: List[ndarray]) -> float:
    diff = flatten_act(act2) - flatten_act(act1)
    return (diff ** 2).mean().item()


def should_stop(act_prev: List[ndarray], act: List[ndarray], eps: float = 1e-4) -> bool:
    diff = flatten_act(act) - flatten_act(act_prev)
    return (np.abs(diff).sum() < eps).item()


def precision(act, perfect_act, threshold=0.5):
    perfect_bool = flatten_act(perfect_act) > 0.5
    positives = flatten_act(act) >= threshold
    n_positives = np.count_nonzero(positives)
    n_true_positives = np.count_nonzero(perfect_bool & positives)
    return (n_true_positives / n_positives) if n_positives else 0.


def recall(act, perfect_act, threshold=0.5):
    perfect_bool = flatten_act(perfect_act) > 0.5
    n_true = np.count_nonzero(perfect_bool)
    positives = flatten_act(act) >= threshold
    n_true_positives = np.count_nonzero(perfect_bool & positives)
    return (n_true_positives / n_true)


def plot_activation_hist(act):
    fig = plt.figure(figsize=(64, 8))
    plots = fig.subplots(1, 7)
    for i in range(7):
        plots[i].hist(act[i].flatten())
    fig.show()


def draw_activation_values(act):
    fig = plt.figure(figsize=(128, 16))
    plots = fig.subplots(1, 7)
    for i in range(7):
        plots[i].imshow(act[i].reshape((10, 10)), vmin=0, vmax=1., cmap='gray')
    plt.show()


def draw_tracks(pos, seg, act, perfect_act, THRESHOLD):
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_zlabel('Z')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    for i in range(7):
        s = seg[i]
        a = act[i]
        a_good = perfect_act[i]
        for ns, jk in enumerate(s.transpose()):
            j, k = jk
            positive = a[ns] > THRESHOLD
            true = a_good[ns] > THRESHOLD
            if positive and true:
                color = 'black'
            elif positive and not true:
                color = 'red'
            elif not positive and true:
                color = 'blue'
            else:
                continue
            xs = [pos[j, 0], pos[k, 0]]
            ys = [pos[j, 1], pos[k, 1]]
            zs = [pos[j, 2], pos[k, 2]]
            ax.plot(xs, ys, zs,
                    color=color,
                    linewidth=1.,
                    marker='.')
    fig.show()


def draw_tracks_projection(pos, seg, act, perfect_act, THRESHOLD):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)

    for i in range(7):
        s = seg[i]
        a = act[i]
        a_good = perfect_act[i]
        for ns, jk in enumerate(s.transpose()):
            j, k = jk
            positive = a[ns] > THRESHOLD
            true = a_good[ns] > THRESHOLD
            if positive and true:
                color = 'black'
            elif positive and not true:
                color = 'red'
            elif not positive and true:
                color = 'blue'
            else:
                continue
            ys = [pos[j, 1], pos[k, 1]]
            zs = [pos[j, 2], pos[k, 2]]
            ax.plot(ys, zs,
                    color=color,
                    linewidth=1.,
                    marker='.')
    fig.show()
