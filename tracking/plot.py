from typing import List

import holoviews as hv
import numpy as np
import pandas as pd


def plot_segments(hits: pd.DataFrame, seg: np.ndarray, kdims: List = ['x', 'y', 'z']) -> hv.Overlay:
    lines = [[hits.loc[a], hits.loc[b]] for a, b in seg]
    return hv.Overlay([
        hv.Path3D(lines, kdims=kdims, label='segments', group='segments'),
        hv.Scatter3D(hits[hits.track == -1], kdims=kdims, label='fake', group='hits'),
        hv.Scatter3D(hits[hits.track != -1], kdims=kdims, label='real', group='hits')
    ])
