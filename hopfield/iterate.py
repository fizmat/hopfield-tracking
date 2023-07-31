from functools import partial
from typing import List, Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numba import njit
from numpy import ndarray
from scipy.sparse import spmatrix, csr_matrix

from datasets import get_hits
from hopfield.energy import energy_gradient
from hopfield.energy.cross import cross_energy_matrix
from hopfield.energy.curvature import find_consecutive_segments, curvature_energy_matrix
from metrics.segments import gen_perfect_act
from metrics.tracks import track_metrics, trackml_score
from segment.candidate import gen_seg_layered
from segment.track import gen_seg_track_layered


def hopfield_history(energy_matrix: spmatrix, temp_curve: np.ndarray, starting_act: np.ndarray,
                     update_act: Callable, bias: float = 0) -> List[np.ndarray]:
    return [act.copy() for act in anneal(energy_matrix, temp_curve, starting_act, update_act, bias)]


def anneal(energy_matrix: spmatrix, temp_curve: np.ndarray, act: np.ndarray,
           update_act: Callable, bias: float = 0.) -> List[np.ndarray]:
    for t in temp_curve:
        update_act(energy_matrix, act, temperature=t, bias=bias)
        yield act


def annealing_curve(t_min: float, t_max: float, cooling_steps: int, rest_steps: int) -> np.ndarray:
    return np.concatenate([
        np.geomspace(t_max, t_min, cooling_steps),
        np.full(rest_steps, t_min)])


def update_act_bulk(energy_matrix: spmatrix, act: ndarray, temperature: float = 1.,
                    learning_rate: float = 1., bias: float = 0.) -> None:
    grad = energy_gradient(energy_matrix, act)
    next_act = 0.5 * (1 + np.tanh((- grad + bias) / temperature))
    act[:] = next_act * learning_rate + act * (1. - learning_rate)


@njit
def _update_act_sequential(indptr: ndarray, indices: ndarray, data: ndarray,
                           act: ndarray, temperature: float = 1.,
                           bias: float = 0.) -> None:
    for i in range(len(act)):
        a, b = indptr[i], indptr[i + 1]
        ind = indices[a:b]
        val = data[a:b]
        grad = 2 * (act[ind] * val).sum()
        act[i] = 0.5 * (1 + np.tanh((- grad + bias) / temperature))


def update_act_sequential(energy_matrix: csr_matrix,
                          act: ndarray, temperature: float = 1.,
                          bias: float = 0.) -> None:
    _update_act_sequential(energy_matrix.indptr, energy_matrix.indices, energy_matrix.data,
                           act, temperature, bias)


def metric_history(event: pd.DataFrame, seg: ndarray, tseg: ndarray, acts: List[ndarray],
                   positives: List[ndarray]) -> pd.DataFrame:
    return pd.DataFrame([track_metrics(event, seg, tseg, act, positive) for act, positive in zip(acts, positives)])


def main():
    from vispy import app
    from vispy.scene import SceneCanvas
    from hopfield.plot import _act_view, _result_view
    from datasets import bman

    event = get_hits('bman', 5)
    event = event[event.event_id == 1].reset_index(drop=True)
    event[['x', 'y', 'z']] /= bman.LAYER_DIST
    alpha = 924.3062112667407
    seg = gen_seg_layered(event)
    crossing_matrix = alpha * cross_energy_matrix(seg)
    curvature_matrix = curvature_energy_matrix(
        pos=event[['x', 'y', 'z']].to_numpy(), seg=seg, pairs=find_consecutive_segments(seg),
        cosine_threshold=0.5658048189789646, curvature_cosine_power=45.37892813716288,
        alpha=alpha, gamma=740.3731323900522, distance_prod_power_in_denominator=0.
    )
    energy_matrix = crossing_matrix + curvature_matrix
    temp_curve = annealing_curve(1., 449.20928874777286, cooling_steps=50, rest_steps=0)
    starting_act = np.full(len(seg), 0.027148322467310068)
    update_act = update_act_sequential
    acts = hopfield_history(energy_matrix, temp_curve, starting_act, update_act, bias=-1.3528558786458618,)
    positives = [act >= 0.5 for act in acts]

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
