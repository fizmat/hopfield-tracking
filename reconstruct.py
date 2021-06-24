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
    next_act = 0.5 * (1 + np.tanh((- grad + bias) / t))
    updated_act = next_act * learning_rate + act * (1. - learning_rate)
    dropout = bernoulli.rvs(dropout_rate)
    return np.where(dropout, act, updated_act)


def precision(act, perfect_act, threshold=0.5):
    perfect_bool = perfect_act > 0.5
    positives = act >= threshold
    n_positives = np.count_nonzero(positives)
    n_true_positives = np.count_nonzero(perfect_bool & positives)
    return (n_true_positives / n_positives) if n_positives else 0.


def recall(act, perfect_act, threshold=0.5):
    perfect_bool = perfect_act > 0.5
    n_true = np.count_nonzero(perfect_bool)
    positives = act >= threshold
    n_true_positives = np.count_nonzero(perfect_bool & positives)
    return n_true_positives / n_true


def draw_tracks(pos: ndarray, seg: ndarray, act: ndarray,
                perfect_act: ndarray, threshold: float):
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_zlabel('Z')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    for ns, jk in enumerate(seg):
        j, k = jk
        positive = act[ns] > threshold
        true = perfect_act[ns] > threshold
        if positive and true:
            color = 'black'
        elif positive and not true:
            color = 'red'
        elif not positive and true:
            color = 'cyan'
        else:
            continue
        xs = [pos[j, 0], pos[k, 0]]
        ys = [pos[j, 1], pos[k, 1]]
        zs = [pos[j, 2], pos[k, 2]]
        ax.plot(xs, ys, zs,
                color=color,
                linewidth=1.,
                marker='')

    ax.scatter(*pos.transpose(), color='k', s=16.)
    fig.show()


def draw_tracks_projection(pos: ndarray, seg: ndarray, act: ndarray,
                           perfect_act: ndarray, threshold: float):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)

    for ns, jk in enumerate(seg):
        j, k = jk
        positive = act[ns] > threshold
        true = perfect_act[ns] > threshold
        if positive and true:
            color = 'black'
        elif positive and not true:
            color = 'red'
        elif not positive and true:
            color = 'cyan'
        else:
            continue
        ys = [pos[j, 1], pos[k, 1]]
        zs = [pos[j, 2], pos[k, 2]]
        ax.plot(ys, zs,
                color=color,
                linewidth=1.,
                marker='')
    ax.scatter(*pos.transpose()[1:], color='k', s=16)
    fig.show()
