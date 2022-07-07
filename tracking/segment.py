from typing import List, Tuple

import numpy as np
import pandas as pd
from numpy import ndarray


def _gen_segments_layer(a: pd.Index, b: pd.Index) -> ndarray:
    return np.stack([x.ravel() for x in np.meshgrid(a, b, indexing='ij')], axis=1)


def gen_segments_all(hits: pd.DataFrame) -> ndarray:
    vert_i_by_layer = [g.index for _, g in hits.groupby('layer')]
    if len(vert_i_by_layer) < 2:
        return np.zeros((0, 2))
    return np.concatenate([_gen_segments_layer(a, b) for a, b in zip(vert_i_by_layer, vert_i_by_layer[1:])])


def gen_seg_track_sequential(event: pd.DataFrame) -> np.ndarray:
    return np.concatenate([np.stack((g.index[:-1], g.index[1:]), axis=-1)
                           for track, g in event.groupby('track')
                           if track >= 0])


def gen_seg_track_layered(hits: pd.DataFrame) -> List[Tuple[int, int]]:
    track_segments = []
    for track, g in hits.groupby('track'):
        if track >= 0:
            for i in range(min(g.layer), max(g.layer)):
                for a in g[g.layer == i].index:
                    for b in g[g.layer == i + 1].index:
                        track_segments.append((a, b))
    return track_segments
