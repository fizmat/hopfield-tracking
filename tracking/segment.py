from typing import List, Tuple

import numpy as np
import pandas as pd
from line_profiler_pycharm import profile
from numpy import ndarray
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix
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


@profile
def seg_drop_same_layer(seg: np.ndarray, event: pd.DataFrame) -> np.ndarray:
    a = event.loc[seg[:, 0], 'layer'].to_numpy()
    b = event.loc[seg[:, 1], 'layer'].to_numpy()
    comp = a != b
    return seg[comp]


@profile
def graph_drop_same_layer(nbr: csr_matrix, event: pd.DataFrame, current_hits: pd.DataFrame) -> csr_matrix:
    indptr = nbr.indptr
    indices = nbr.indices
    data = nbr.data
    new_indptr = np.zeros_like(indptr)
    new_indices = np.empty_like(indices)
    new_data = np.empty_like(data)
    for i in range(nbr.shape[0]):
        start = indptr[i]
        end = indptr[i + 1]
        new_start = new_indptr[i]
        jj = indices[start:end]
        comparison = current_hits.layer.to_numpy()[i] != event.layer.to_numpy()[jj]
        new_end = new_start + comparison.sum()
        new_indptr[i + 1] = new_end
        new_data[new_start:new_end] = data[start:end][comparison]
        new_indices[new_start:new_end] = indices[start:end][comparison]
    new_graph = csr_matrix((new_data[:new_indptr[-1]], new_indices[:new_indptr[-1]], new_indptr), nbr.shape)
    return new_graph


@profile
def nbr_stat_block(current_hits: pd.DataFrame, neighbors_model: NearestNeighbors,
                   distances: np.ndarray, event: pd.DataFrame) -> List[Tuple[float, float, float]]:
    records = []
    nbr = neighbors_model.radius_neighbors_graph(current_hits[['x', 'y', 'z']],
                                                 radius=distances.max(initial=0),
                                                 mode='distance')
    nbr_diff_layer = graph_drop_same_layer(nbr, event, current_hits)
    for r in distances:
        n_seg_all = nbr.nnz - (nbr > r).nnz - nbr.shape[0]  # substract the hit being its own neighbor
        n_seg_diff_layer = nbr_diff_layer.nnz - (nbr_diff_layer > r).nnz
        records.append((r, n_seg_all, n_seg_diff_layer))
    return records


@profile
def build_segment_neighbor(event, nbr):
    seg_list = []
    for i, ends in enumerate(nbr):
        starts = np.full_like(ends, event.index[i])
        neighbor_seg = np.stack((starts, ends), axis=-1)
        seg_list.append(neighbor_seg)
    seg = np.concatenate(seg_list, axis=0)
    return seg


@profile
def stat_seg_neighbors_event_r(neighbors_model: NearestNeighbors, distances: np.ndarray,
                               event: pd.DataFrame) -> pd.Series:
    records = [record for _, g in event.groupby('layer') for record in
               nbr_stat_block(g, neighbors_model, distances, event)]
    return pd.DataFrame(records, columns=['r', 'seg_all', 'seg_diff_level']).groupby(
        'r').sum().reset_index()


@profile
def stat_seg_neighbors(events: pd.DataFrame) -> pd.DataFrame:
    stat_blocks = []
    distances = np.linspace(300, 3000, 10)

    for ei, event in events.groupby(by='event_id'):
        event = event.reset_index(drop=True)
        neighbors_model = NearestNeighbors().fit(event[['x', 'y', 'z']])
        stats = stat_seg_neighbors_event_r(neighbors_model, distances, event)
        stats['event'] = ei
        stat_blocks.append(stats)
    return pd.concat(stat_blocks, ignore_index=True)


def _profile():
    from datasets import get_hits
    event = get_hits('trackml_volume', 1)
    print(stat_seg_neighbors(event))


if __name__ == '__main__':
    _profile()
