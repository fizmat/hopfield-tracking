from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy.sparse import spmatrix

from datasets import get_hits
from hopfield.energy import energy_gradient
from hopfield.energy.cross import cross_energy_matrix
from hopfield.energy.curvature import segment_adjacent_pairs, curvature_energy_matrix
from metrics.segments import gen_perfect_act
from metrics.tracks import track_metrics
from segment.candidate import gen_seg_layered
from segment.track import gen_seg_track_layered


def hopfield_history(energy_matrix: spmatrix, temp_curve: np.ndarray, starting_act: np.ndarray,
                     learning_rate: float = 1, bias: float = 0) -> List[np.ndarray]:
    return [act.copy() for act in anneal(energy_matrix, temp_curve, starting_act, learning_rate, bias)]


def anneal(energy_matrix: spmatrix, temp_curve: np.ndarray, act: np.ndarray,
           learning_rate: float = 1, bias: float = 0) -> List[np.ndarray]:
    for t in temp_curve:
        grad = energy_gradient(energy_matrix, act)
        update_layer_grad(act, grad, t, learning_rate, bias)
        yield act


def annealing_curve(t_min: float, t_max: float, cooling_steps: int, rest_steps: int) -> np.ndarray:
    return np.concatenate([
        np.geomspace(t_max, t_min, cooling_steps),
        np.full(rest_steps, t_min)])


def update_layer_grad(act: ndarray, grad: ndarray, t: float,
                      learning_rate: float = 1., bias: float = 0.) -> None:
    n = len(act)
    next_act = 0.5 * (1 + np.tanh((- grad + bias) / t))
    act[:] = next_act * learning_rate + act * (1. - learning_rate)


def run(event: pd.DataFrame,
        alpha=1., gamma=1.,
        cosine_min_rewarded=.0, cosine_min_allowed=-1., cosine_power=1.0,
        distance_op='sum', distance_power=1.,
        t_min=1.0, t_max=1.0, cooling_steps=50, rest_steps=50,
        initial_act=0.5, learning_rate=1.0, bias=0., threshold=0.5):
    pos = event[['x', 'y', 'z']].to_numpy()
    seg = gen_seg_layered(event)
    pairs = segment_adjacent_pairs(seg)
    crossing_matrix = cross_energy_matrix(seg)
    curvature_matrix = curvature_energy_matrix(
        pos, seg, pairs, alpha=alpha, gamma=gamma,
        cosine_threshold=cosine_min_rewarded, cosine_min_allowed=cosine_min_allowed,
        curvature_cosine_power=cosine_power,
        do_sum_r=distance_op == 'sum', distance_prod_power_in_denominator=distance_power
    )
    energy_matrix = crossing_matrix + curvature_matrix
    temp_curve = annealing_curve(t_min, t_max, cooling_steps, rest_steps)
    starting_act = np.full(len(seg), initial_act)
    acts = hopfield_history(energy_matrix, temp_curve, starting_act,
                            learning_rate=learning_rate, bias=bias)
    positive = [act >= threshold for act in acts]
    return seg, acts, positive


def metric_history(event: pd.DataFrame, seg: ndarray, tseg: ndarray, acts: List[ndarray],
                   positives: List[ndarray]) -> pd.DataFrame:
    return pd.DataFrame([track_metrics(event, seg, tseg, act, positive) for act, positive in zip(acts, positives)])


def main():
    from vispy import app
    from vispy.scene import SceneCanvas
    from hopfield.plot import _act_view, _result_view

    event = get_hits('spdsim', 1)
    seg, acts, positives = run(event, alpha=1, gamma=2,
                               cosine_power=3, cosine_min_allowed=-2, cosine_min_rewarded=0.8,
                               distance_op='sum', distance_power=0.,
                               t_min=1, t_max=10, cooling_steps=100, rest_steps=10,
                               initial_act=0.5, learning_rate=0.1, bias=-2)

    canvas = SceneCanvas(bgcolor='white', size=(1024, 768))
    grid = canvas.central_widget.add_grid()
    tseg = gen_seg_track_layered(event)
    perfect_act = gen_perfect_act(seg, tseg)
    act = acts[-1]
    metrics = metric_history(event, seg, tseg, acts, positives)
    metrics.plot()
    plt.show()
    act_view = _act_view(event, seg, act)
    grid.add_widget(act_view)
    grid.add_widget(_result_view(event, seg, act, perfect_act, positive=positives[-1], camera=act_view.camera))
    canvas.show()
    app.run()


if __name__ == '__main__':
    main()
