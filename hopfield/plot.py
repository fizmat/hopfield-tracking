from typing import Tuple, List, Dict, Any, Iterable, Union

import numpy as np
import pandas as pd
from vispy.color import colormap
from vispy.scene import ViewBox, visuals, SceneCanvas

from metrics.segments import gen_perfect_act
from segment.candidate import gen_seg_layered


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
    cmap = colormap.MatplotlibColormap('viridis')
    colors = cmap.map(np.stack([act, act], axis=1).flatten())
    seg_lines.set_data(seg_hits[kdims].to_numpy(), color=colors)
    view.add(seg_lines)
    visuals.XYZAxis(parent=view.scene)
    view.camera = camera
    return view


def main():
    from datasets import get_hits
    from vispy import app

    canvas = SceneCanvas(bgcolor='white', size=(1024, 768))
    grid = canvas.central_widget.add_grid()

    event = get_hits('bman', 1)
    seg = gen_seg_layered(event)
    act = gen_perfect_act(event, seg)
    grid.add_widget(_act_view(event, seg, act))
    canvas.show()
    app.run()


if __name__ == '__main__':
    main()
