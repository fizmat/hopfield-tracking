from typing import Tuple, List, Dict, Any, Iterable

import numpy as np
import pandas as pd
from vispy.color import colormap
from vispy.scene import ViewBox, visuals, SceneCanvas

from metrics.segments import gen_perfect_act
from segment.candidate import gen_seg_layered
from segment.track import gen_seg_track_layered


def make_tracks_3d(
        pos: np.ndarray, seg: np.ndarray, act: np.ndarray, perfect_act: np.ndarray, threshold: float
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


def _act_view(event: pd.DataFrame, seg: np.ndarray, act: np.ndarray,
              kdims: Iterable = ('x', 'y', 'z'), camera='turntable') -> ViewBox:
    kdims = list(kdims)
    view = ViewBox(border_color='black')
    seg_lines = visuals.Line(connect='segments')
    seg_hits = event.loc[seg.flatten()]
    cmap = colormap.CubeHelixColormap()
    colors = cmap.map(np.stack([act, act], axis=1).flatten())
    seg_lines.set_data(seg_hits[kdims].to_numpy(), color=colors)
    view.add(seg_lines)
    visuals.XYZAxis(parent=view.scene)
    view.camera = camera
    return view


def _result_view(event: pd.DataFrame, seg: np.ndarray, act: np.ndarray, perfect_act: np.ndarray,
                 positive: np.ndarray, kdims: Iterable = ('x', 'y', 'z'), camera='turntable') -> ViewBox:
    kdims = list(kdims)
    view = ViewBox(border_color='black')
    is_true = perfect_act > 0.5
    draw_first = np.logical_or(is_true, positive)
    for do_draw in draw_first, np.logical_not(draw_first):
        seg_lines = visuals.Line(connect='segments')
        seg_hits = event.loc[seg[do_draw].flatten()]
        trueness = np.stack([is_true[do_draw], is_true[do_draw]], axis=1).flatten().astype(int)
        positiveness = np.stack([positive[do_draw], positive[do_draw]], axis=1).flatten().astype(int)
        color_map = np.array([[(.5, .5, .5, 0.2), (.8, 0, 0, 1)],
                              [(0, 0, 1, 1), (0, .7, 0, 1)]])
        colors = color_map[trueness, positiveness]
        seg_lines.set_data(seg_hits[kdims].to_numpy(), color=colors)
        view.add(seg_lines)
    visuals.XYZAxis(parent=view.scene)
    view.camera = camera
    return view


def plot_result(event: pd.DataFrame, seg: np.ndarray, act: np.ndarray, perfect_act: np.ndarray,
                positive: np.ndarray, kdims: Iterable = ('x', 'y', 'z')):
    canvas = SceneCanvas(bgcolor='white', size=(1024, 768))
    grid = canvas.central_widget.add_grid()
    act_view = _act_view(event, seg, act, kdims)
    grid.add_widget(act_view)
    grid.add_widget(_result_view(event, seg, act, perfect_act, positive, kdims, camera=act_view.camera))
    return canvas


def main():
    from datasets import get_hits
    from vispy import app
    event = get_hits('simple', 1)
    seg = gen_seg_layered(event)
    tseg = gen_seg_track_layered(event)
    perfect_act = gen_perfect_act(seg, tseg)
    act = np.random.random(perfect_act.shape)
    canvas = plot_result(event, seg, act, perfect_act, positive=act > 0.9)
    canvas.show()
    app.run()


if __name__ == '__main__':
    main()
