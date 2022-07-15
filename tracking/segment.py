import numpy as np
import pandas as pd
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


def create_neighbor_with_radius_not_same_layer(nbr, current_hits, events):
    nbrn = np.empty_like(nbr, dtype='object')

    for i, neighbor_array in enumerate(nbr):
        comparison_layer = current_hits.layer.iloc[i] != events.layer.iloc[neighbor_array]
        # compariso_volume = current_hits.volume_id.iloc[i] != events.volume_id.iloc[neighbor_array]
        comparison = comparison_layer
        nbrn[i] = neighbor_array[comparison.values]

    return (nbrn)


def neighbor_row(hit, neighbors, radius, events):
    current_hits = hit.to_frame().T
    nbr = neighbors.radius_neighbors(current_hits[['x', 'y', 'z']], radius=radius, return_distance=False)
    nbrn = create_neighbor_with_radius_not_same_layer(nbr, current_hits, events)
    lnbr = len(nbr[0])-1
    lnbrn = len(nbrn[0])
    return pd.Series([lnbr,lnbrn],index = ('lnbr','lnbrn'))


def build_segment_neighbor(event, nbrn):
    seg_list = []
    for i, neighbor_array in enumerate(nbrn):
        hit_array = np.full_like(neighbor_array, event.index[i])
        neighbor_seg = np.stack((hit_array, neighbor_array), axis=-1)
        seg_list.append(neighbor_seg)
    seg = np.concatenate(seg_list, axis=0)
    return seg


def stat_seg_neighbors(events: pd.DataFrame) -> pd.DataFrame:
    records = []

    for radius in range(100, 201, 50):
        all_segments = 0
        neighbor_segments_not_level = 0

        for ei, ev in events.groupby(by='event_id'):
            event = ev.reset_index(drop=True)
            neighbors = NearestNeighbors().fit(event[['x', 'y', 'z']])

            quantity_neighbor_hit = event.apply(neighbor_row, args=(neighbors, radius, event), axis=1)
            lnbr, lnbrn = quantity_neighbor_hit.sum()

            all_segments += lnbr
            neighbor_segments_not_level += lnbrn

        records.append([radius, all_segments, neighbor_segments_not_level])
    return pd.DataFrame(data=records, columns=['r','all_segments','neighbor_segments_not_level']).set_index('r')


def _profile():
    from datasets import get_hits
    event = get_hits('trackml_volume', 1)
    print(stat_seg_neighbors(event.iloc[:1000]))


if __name__ == '__main__':
    _profile()
