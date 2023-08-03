from typing import Iterable, Union, Tuple

import holoviews as hv
import numpy as np
import pandas as pd
from holoviews import opts
from vispy.color import ColorArray, Color, colormap
from vispy.scene import ViewBox, visuals, SceneCanvas


def _hits_view(event: pd.DataFrame, kdims: Iterable = ('x', 'y', 'z'),
               color: Union[Color, ColorArray] = 'black',
               symbol='o', size=5, camera='turntable') -> ViewBox:
    kdims = list(kdims)
    view = ViewBox(border_color='black')
    scatter = visuals.Markers()
    scatter.set_data(event[kdims].to_numpy(), edge_color=color, face_color=color, symbol=symbol, size=size)
    view.add(scatter)
    visuals.XYZAxis(parent=view.scene)
    view.camera = camera
    return view


def _seg_view(event: pd.DataFrame, seg: np.ndarray, kdims: Iterable = ('x', 'y', 'z'),
              color: Union[Color, ColorArray] = 'black', camera='turntable') -> ViewBox:
    kdims = list(kdims)
    view = ViewBox(border_color='black')
    seg_lines = visuals.Line(connect='segments')
    seg_hits = event.loc[seg.flatten()]
    seg_lines.set_data(seg_hits[kdims].to_numpy(), color=color)
    view.add(seg_lines)
    visuals.XYZAxis(parent=view.scene)
    view.camera = camera
    return view


def _seg_tseg_view(event: pd.DataFrame, seg: np.ndarray, tseg: np.ndarray,
                   kdims: Iterable = ('x', 'y', 'z'), camera='turntable') -> ViewBox:
    kdims = list(kdims)
    view = ViewBox(border_color='black')
    tseg_lines = visuals.Line(connect='segments', width=2, color='black')
    seg_lines = visuals.Line(connect='segments', color=(.2, .2, .2, 0.4))
    seg_hits = event.loc[seg.flatten()]
    tseg_hits = event.loc[tseg.flatten()]
    tseg_lines.set_data(tseg_hits[kdims].to_numpy())
    seg_lines.set_data(seg_hits[kdims].to_numpy())
    view.add(tseg_lines)
    view.add(seg_lines)
    visuals.XYZAxis(parent=view.scene)
    view.camera = camera
    return view


def plot_event(event: pd.DataFrame, seg: np.ndarray = None, kdims: Iterable = ('x', 'y', 'z'),
               fig_size: Tuple[int, int] = (1024, 768), camera='turntable', black_white=False) -> SceneCanvas:
    kdims = list(kdims)
    canvas = SceneCanvas(bgcolor='white', size=fig_size)
    grid = canvas.central_widget.add_grid()

    if black_white:
        color = 'black'
        symbol = np.where((event.track.to_numpy() == -1), 'x', 'o')
        size = 4
    else:
        color = np.where((event.track.to_numpy() == -1)[..., np.newaxis], [.8, .1, .1], [.1, .8, .1])
        symbol = 'o'
        size = 5
    fakes_view = _hits_view(event, kdims, color, symbol, size=size, camera=camera)
    grid.add_widget(fakes_view)

    event = event[event.track != -1]
    tracks = event.track.unique()
    track_map = {t: i for i, t in enumerate(tracks)}
    if black_white:
        symbols = ['disc', 'x', 'cross', 'triangle_down', 'star', 'ring', 'arrow', 'clobber', 'square', 'diamond',
                   'vbar', 'hbar', 'tailed_arrow', 'triangle_up']
        color = 'black'
        symbol = event.track.map(lambda t: symbols[track_map[t] % len(symbols)])
        size = 4
    else:
        cmap = colormap.MatplotlibColormap('tab20')
        color = cmap.map(event.track.map(track_map) / (len(tracks) - 1))
        symbol = 'o'
        size = 5

    track_view = _hits_view(event, kdims, color, symbol, size, fakes_view.camera)
    grid.add_widget(track_view)

    if seg is not None:
        seg_view = _seg_view(event, seg, kdims, camera=fakes_view.camera)
        grid.add_widget(seg_view)
    return canvas


def plot_seg_diff(event: pd.DataFrame, seg1: np.ndarray, seg2: np.ndarray, kdims: Iterable = ('x', 'y', 'z'),
                  fig_size: Tuple[int, int] = (1024, 768)) -> SceneCanvas:
    kdims = list(kdims)
    canvas = SceneCanvas(bgcolor='white', size=fig_size)
    grid = canvas.central_widget.add_grid()
    view1 = _seg_view(event, seg1, kdims)
    grid.add_widget(view1)

    s1, s2 = {tuple(pair) for pair in seg1}, {tuple(pair) for pair in seg2}
    red = np.array(list(s1 - s2))
    red_view = _seg_view(event, red, kdims, 'red', view1.camera)
    grid.add_widget(red_view)

    green = np.array(list(s2 - s1))
    green_view = _seg_view(event, green, kdims, 'green', view1.camera)
    grid.add_widget(green_view, row=1, col=0)

    view2 = _seg_view(event, seg2, kdims, camera=view1.camera)
    grid.add_widget(view2, row=1, col=1)
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


def main():
    from datasets import get_hits
    from segment.track import gen_seg_track_layered, gen_seg_track_sequential
    from vispy import app

    event = get_hits('bman', 1)
    plot_event(event, gen_seg_track_sequential(event)).show()
    app.run()
    plot_seg_diff(event, gen_seg_track_layered(event), gen_seg_track_sequential(event)).show()
    app.run()


if __name__ == '__main__':
    main()
