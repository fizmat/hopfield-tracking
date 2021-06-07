from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor


def annealing_curve(t_min, t_max, cooling_steps, rest_steps):
    return np.concatenate([
        np.geomspace(t_max, t_min, cooling_steps),
        np.full(rest_steps, t_min)])


def update_layer(act: Tensor, t: float, dropout_rate: float = 0., learning_rate: float = 1.) -> Tensor:
    next_act = 0.5 * (1 + torch.tanh(- act.grad / t))
    updated_act = next_act * learning_rate + act * (1. - learning_rate)
    dropout = torch.bernoulli(torch.full_like(act, dropout_rate)).bool()
    dropped_act = torch.where(dropout, act, updated_act)
    result = dropped_act.clone().detach().requires_grad_(True)
    return result


def flatten_act(act: List[Tensor]) -> Tensor:
    return torch.cat(tuple(a.flatten() for a in act))


def mean_act(act: List[Tensor]) -> float:
    return flatten_act(act).mean().item()


def dist_act(act1: List[Tensor], act2: List[Tensor]) -> float:
    diff = flatten_act(act2) - flatten_act(act1)
    return (diff ** 2).mean().item()


def should_stop(act_prev: List[Tensor], act: List[Tensor], eps: float = 1e-4) -> bool:
    diff = flatten_act(act) - flatten_act(act_prev)
    return (diff.abs().sum() < eps).item()


def precision(act, perfect_act, threshold=0.5):
    perfect_bool = flatten_act(perfect_act) > 0.5
    positives = flatten_act(act) >= threshold
    n_positives = torch.count_nonzero(positives)
    n_true_positives = torch.count_nonzero(perfect_bool & positives)
    return (n_true_positives / n_positives).item()


def recall(act, perfect_act, threshold=0.5):
    perfect_bool = flatten_act(perfect_act) > 0.5
    n_true = torch.count_nonzero(perfect_bool)
    positives = flatten_act(act) >= threshold
    n_true_positives = torch.count_nonzero(perfect_bool & positives)
    return (n_true_positives / n_true).item()


def plot_activation_hist(act):
    fig = plt.figure(figsize=(64, 8))
    plots = fig.subplots(1, 7)
    for i in range(7):
        plots[i].hist(act[i].flatten().cpu().detach().numpy())
    fig.show()


def draw_activation_values(act):
    fig = plt.figure(figsize=(128, 16))
    plots = fig.subplots(1, 7)
    for i in range(7):
        plots[i].imshow(act[i].cpu().detach().numpy(), vmin=0, vmax=1., cmap='gray')
    plt.show()


def draw_tracks(pos, act, perfect_act, THRESHOLD):
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_zlabel('Z')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    for i in range(7):
        p1 = pos[i].cpu()
        p2 = pos[i + 1].cpu()
        a = act[i].cpu()
        a_good = perfect_act[i].cpu()
        for j in range(len(p1)):
            for k in range(len(p2)):
                positive = a[j, k] > THRESHOLD
                true = a_good[j, k] > THRESHOLD
                if positive and true:
                    color = 'black'
                elif positive and not true:
                    color = 'red'
                elif not positive and true:
                    color = 'blue'
                else:
                    continue
                xs = [p1[j, 0], p2[k, 0]]
                ys = [p1[j, 1], p2[k, 1]]
                zs = [p1[j, 2], p2[k, 2]]
                ax.plot(xs, ys, zs,
                        color=color,
                        linewidth=1.,
                        marker='.')
    fig.show()


def draw_tracks_projection(pos, act, perfect_act, THRESHOLD):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)

    for i in range(7):
        p1 = pos[i].cpu()
        p2 = pos[i + 1].cpu()
        a = act[i].cpu()
        a_good = perfect_act[i].cpu()
        for j in range(len(p1)):
            for k in range(len(p2)):
                positive = a[j, k] > THRESHOLD
                true = a_good[j, k] > THRESHOLD
                if positive and true:
                    color = 'black'
                elif positive and not true:
                    color = 'red'
                elif not positive and true:
                    color = 'blue'
                else:
                    continue
                ys = [p1[j, 1], p2[k, 1]]
                zs = [p1[j, 2], p2[k, 2]]
                ax.plot(ys, zs,
                        color=color,
                        linewidth=1.,
                        marker='.')
    fig.show()
