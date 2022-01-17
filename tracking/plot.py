from typing import Iterable

import holoviews as hv
import numpy as np
import pandas as pd
from holoviews import opts


def plot_segments(hits: pd.DataFrame, seg: np.ndarray, kdims: Iterable = ('x', 'y', 'z')) -> hv.Overlay:
    kdims = list(kdims)
    lines = [[hits.loc[a], hits.loc[b]] for a, b in seg]
    return hv.Overlay([
        hv.Path3D(lines, kdims=kdims, label='segments', group='segments'),
        hv.Scatter3D(hits[hits.track == -1], kdims=kdims, label='fake', group='hits'),
        hv.Scatter3D(hits[hits.track != -1], kdims=kdims, label='real', group='hits')
    ])


def plot_tracks(hits: pd.DataFrame, kdims: Iterable = ('x', 'y', 'z')) -> hv.Overlay:
    kdims = list(kdims)
    tracks = {track: group.sort_values('track') for track, group in hits[hits.track != -1].groupby('track')}
    return hv.Overlay(
        [hv.Path3D(hits, kdims=kdims, label=str(track), group='tracks') for track, hits in tracks.items()]
    )


def plot_segments_matplotlib(hits: pd.DataFrame, seg: np.ndarray, kdims: Iterable = ('x', 'y', 'z')):
    hv.extension('matplotlib')
    return plot_segments(hits, seg, kdims).opts(
        opts.Scatter3D('Hits.Real', c='green'),
        opts.Scatter3D('Hits.Fake', c='black'),
        opts.Path3D('Segments.Segments', color='black', alpha=0.2),
        opts.Overlay(fig_size=400),
    )


def plot_segments_plotly(hits: pd.DataFrame, seg: np.ndarray, kdims: Iterable = ('x', 'y', 'z')):
    hv.extension('plotly')
    return plot_segments(hits, seg, kdims).opts(
        opts.Scatter3D('Hits.Real', color='green'),
        opts.Scatter3D('Hits.Fake', color='black'),
        opts.Path3D('Segments.Segments', color='rgba(0, 0, 0, 0.3)'),
        opts.Overlay(width=800, height=800),
    )
