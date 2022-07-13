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


def gen_seg_track_layered(event: pd.DataFrame) -> np.ndarray:
    track_segments = []
    for track, g in event[event.track >= 0].groupby('track'):
        layers = g.groupby('layer').groups
        for layer, starts in layers.items():
            for b in layers.get(layer + 1, ()):
                for a in starts:
                    track_segments.append((a, b))
    return np.array(track_segments)


def _profile():
    from datasets import get_hits_trackml_one_event, get_hits_trackml_one_event_by_volume
    event = get_hits_trackml_one_event_by_volume()
    gen_segments_all(event)
    gen_seg_track_sequential(event)
    gen_seg_track_layered(event)


if __name__ == '__main__':
    _profile()
