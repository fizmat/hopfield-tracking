from typing import Iterable, Union, Tuple

import holoviews as hv
import numpy as np
import pandas as pd
from holoviews import opts
from vispy.color import ColorArray, Color, colormap
from vispy.scene import ViewBox, visuals, SceneCanvas


def _hits_view(event: pd.DataFrame, kdims: Iterable = ('x', 'y', 'z'),
               color: Union[Color, ColorArray] = 'black') -> ViewBox:
    view = ViewBox(border_color='black')
    scatter = visuals.Markers()
    scatter.set_data(event[kdims].to_numpy(), edge_color=color, face_color=color, size=3)
    view.add(scatter)
    visuals.XYZAxis(parent=view.scene)
    view.camera = 'turntable'
    return view


def _seg_view(event: pd.DataFrame, seg: np.ndarray, kdims: Iterable = ('x', 'y', 'z'),
              color: Union[Color, ColorArray] = 'black') -> ViewBox:
    view = ViewBox(border_color='black')
    seg_lines = visuals.Line(connect='segments')
    seg_hits = event.loc[np.concatenate(seg)]
    seg_lines.set_data(seg_hits[kdims].to_numpy(), color=color)
    view.add(seg_lines)
    visuals.XYZAxis(parent=view.scene)
    view.camera = 'turntable'
    return view


def plot_event(event: pd.DataFrame, seg: np.ndarray = None, kdims: Iterable = ('x', 'y', 'z'),
               fig_size: Tuple[int, int] = (1024, 768)) -> SceneCanvas:
    kdims = list(kdims)
    canvas = SceneCanvas(bgcolor='white', size=fig_size)
    grid = canvas.central_widget.add_grid()

    color = np.where((event.track.to_numpy() == -1)[..., np.newaxis], [.8, .1, .1], [.1, .8, .1])
    fakes_view = _hits_view(event, kdims, color)
    grid.add_widget(fakes_view)

    cmap = colormap.MatplotlibColormap('tab20')
    event = event[event.track != -1]
    track_enum = {t: np.random.rand() for t in event.track.unique()}
    color = cmap.map(event.track.map(track_enum))
    track_view = _hits_view(event, kdims, color)
    grid.add_widget(track_view)
    fakes_view.camera.link(track_view.camera)

    if seg is not None:
        seg_view = _seg_view(event, seg, kdims)
        grid.add_widget(seg_view)
        seg_view.camera.link(fakes_view.camera)
    return canvas


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
