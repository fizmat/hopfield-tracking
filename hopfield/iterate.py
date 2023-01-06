from typing import List

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.sparse import csr_matrix, spmatrix

from datasets import get_hits
from hopfield.energy import energy_gradient
from hopfield.energy.cross import cross_energy_matrix
from hopfield.energy.curvature import segment_adjacent_pairs, curvature_energy_matrix
from metrics.tracks import trackml_score
from segment.candidate import gen_seg_layered
from segment.track import gen_seg_track_layered

def construct_energy_matrix(config: Dict, pos: np.ndarray, seg: np.ndarray
                            ) -> Tuple[csr_matrix, csr_matrix, csr_matrix]:
    pairs = segment_adjacent_pairs(seg)
    crossing_matrix = cross_energy_matrix(seg)
    curvature_matrix = curvature_energy_matrix(pos, seg, pairs,
                                               config['cosine_power'], config['cosine_min_rewarded'],
                                               config['distance_power'])
    return crossing_matrix + curvature_matrix, crossing_matrix, curvature_matrix

def hopfield_history(energy_matrix: spmatrix, temp_curve: np.ndarray, starting_act: np.ndarray,
                     dropout: float = 0, learning_rate: float = 1, bias: float = 0) -> List[np.ndarray]:
    act = starting_act.copy()
    acts = [act.copy()]
    for i, t in enumerate(temp_curve):
        grad = energy_gradient(energy_matrix, act)
        update_layer_grad(act, grad, t, dropout, learning_rate, bias)
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


def main():
    from vispy import app
    from vispy.scene import SceneCanvas
    from metrics.segments import gen_perfect_act
    from hopfield.plot import _act_view, _result_view

    event = get_hits('spdsim', 1)
    pos = event[['x', 'y', 'z']].to_numpy()
    seg = gen_seg_layered(event)
    pairs = segment_adjacent_pairs(seg)
    crossing_matrix = cross_energy_matrix(seg)
    curvature_matrix = curvature_energy_matrix(pos, seg, pairs, alpha=17, gamma=14,
                                               cosine_threshold=0.8, cosine_min_allowed=-0.8, curvature_cosine_power=9,
                                               distance_prod_power_in_denominator=1.)
    energy_matrix = crossing_matrix + curvature_matrix
    temp_curve = annealing_curve(1., 11.5, 20, 80)
    starting_act = np.full(len(seg), 0.92)
    acts = hopfield_history(energy_matrix, temp_curve, starting_act, learning_rate=0.165, bias=1.1)

    canvas = SceneCanvas(bgcolor='white', size=(1024, 768))
    grid = canvas.central_widget.add_grid()
    tseg = gen_seg_track_layered(event)
    perfect_act = gen_perfect_act(seg, tseg)
    act = acts[-1]
    df = pd.DataFrame({'trackml_score': [trackml_score(event, seg, a) for a in acts]})
    df.plot()
    plt.show()
    act_view = _act_view(event, seg, act)
    grid.add_widget(act_view)
    grid.add_widget(_result_view(event, seg, act, perfect_act, camera=act_view.camera))
    canvas.show()
    app.run()


if __name__ == '__main__':
    main()
