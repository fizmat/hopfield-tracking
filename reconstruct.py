from typing import List

import numpy as np
import torch
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
