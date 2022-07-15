import numpy as np
import pandas as pd
from line_profiler_pycharm import profile
from numpy import ndarray
from numpy.typing import ArrayLike
from sklearn.neighbors import NearestNeighbors


def gen_seg_all(event: pd.DataFrame) -> ndarray:
    df = event.reset_index()
    pairs = df.merge(df, how='cross')
    return pairs[pairs.index_x < pairs.index_y][['index_x', 'index_y']].to_numpy()


def _gen_seg_one_layer(a: ArrayLike, b: ArrayLike) -> ndarray:
    return np.stack([x.ravel() for x in np.meshgrid(a, b, indexing='ij')], axis=1)


def gen_seg_layered(event: pd.DataFrame) -> ndarray:
    vert_i_by_layer = [g.index for _, g in event.groupby('layer')]
    if len(vert_i_by_layer) < 2:
        return np.zeros((0, 2))
    return np.concatenate([_gen_seg_one_layer(a, b) for a, b in zip(vert_i_by_layer, vert_i_by_layer[1:])])


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


def seg_drop_same_layer(seg: np.ndarray, event: pd.DataFrame) -> np.ndarray:
    return seg[event.loc[seg[:, 0], 'layer'] != event.loc[seg[:, 0]], 'layer']


def nbr_drop_same_layer(nbr, current_hits, event):
    nbrn = np.empty_like(nbr, dtype='object')
    for i, neighbor_array in enumerate(nbr):
        comparison: pd.Series = current_hits.layer.iloc[i] != event.layer.iloc[neighbor_array]
        nbrn[i] = neighbor_array[comparison.to_numpy()]
    return nbrn


def neighbor_row(hit, neighbors, radius, event):
    current_hits = hit.to_frame().T
    nbr = neighbors.radius_neighbors(current_hits[['x', 'y', 'z']], radius=radius, return_distance=False)
    nbrn = nbr_drop_same_layer(nbr, current_hits, event)
    lnbr = len(nbr[0]) - 1  # -1 for the hit being its own neighbor
    lnbrn = len(nbrn[0])  # [0] because there is only one hit, so one element in nbr
    return pd.Series([lnbr, lnbrn], index=('all_segments', 'segments_not_same_level'))


def build_segment_neighbor(event, nbr):
    seg_list = []
    for i, ends in enumerate(nbr):
        starts = np.full_like(ends, event.index[i])
        neighbor_seg = np.stack((starts, ends), axis=-1)
        seg_list.append(neighbor_seg)
    seg = np.concatenate(seg_list, axis=0)
    return seg


def stat_seg_neighbors_event_r(neighbors_model: NearestNeighbors, radius: float, event: pd.DataFrame) -> pd.Series:
    return event.apply(neighbor_row, args=(neighbors_model, radius, event), axis=1).sum()


@profile
def stat_seg_neighbors(events: pd.DataFrame) -> pd.DataFrame:
    records = []
    distances = np.linspace(100, 200, 3)

    for ei, event in events.groupby(by='event_id'):
        event = event.reset_index(drop=True)
        neighbors_model = NearestNeighbors().fit(event[['x', 'y', 'z']])
        for r in distances:
            lnbr, lnbrn = stat_seg_neighbors_event_r(neighbors_model, r, event)
            records.append((r, ei, lnbr, lnbrn))
    stats = pd.DataFrame(records, columns=['r', 'event', 'all_segments', 'segments_not_same_level'])
    return stats.drop(columns='event').groupby('r').sum()


def _profile():
    from datasets import get_hits
    event = get_hits('trackml_volume', 1)
    print(stat_seg_neighbors(event.iloc[:1000]))


if __name__ == '__main__':
    _profile()
