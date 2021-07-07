from typing import Tuple, Dict, List, Any

import holoviews as hv
import numpy as np
import pandas as pd
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


def make_tracks_3d(
        pos: ndarray, seg: ndarray, act: ndarray, perfect_act: ndarray, threshold: float
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    segment_paths = [
        {'x': xyz[:, 0], 'y': xyz[:, 1], 'z': xyz[:, 2],
         'act': a, 'perfect_act': pa,
         'positive': a > threshold,
         'true': pa > threshold,
         } for xyz, a, pa in zip(pos[seg], act, perfect_act)
    ]
    tp = [p for p in segment_paths if p['true'] and p['positive']]
    fp = [p for p in segment_paths if not p['true'] and p['positive']]
    tn = [p for p in segment_paths if not p['true'] and not p['positive']]
    fn = [p for p in segment_paths if p['true'] and not p['positive']]
    return tp, fp, tn, fn
