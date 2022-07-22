import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.typing import ArrayLike


def gen_seg_all(event: pd.DataFrame) -> ndarray:
    seg = np.stack([x.ravel() for x in np.meshgrid(event.index, event.index)], axis=1)
    return seg[seg[:, 0] < seg[:, 1]]


def _gen_seg_one_layer(a: ArrayLike, b: ArrayLike) -> ndarray:
    return np.stack([x.ravel() for x in np.meshgrid(a, b, indexing='ij')], axis=1)


def gen_seg_layered(event: pd.DataFrame) -> ndarray:
    vert_i_by_layer = [g.index for _, g in event.groupby('layer')]
    if len(vert_i_by_layer) < 2:
        return np.zeros((0, 2))
    return np.concatenate([_gen_seg_one_layer(a, b) for a, b in zip(vert_i_by_layer, vert_i_by_layer[1:])])
