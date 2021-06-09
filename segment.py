from itertools import islice
from typing import Iterable, Tuple, Union, List

import numpy as np
import pandas as pd
from numpy import ndarray
from scipy import sparse
from scipy.sparse import coo_matrix, spmatrix, csr_matrix, block_diag


def gen_segments_layer(a: pd.Index, b: pd.Index) -> ndarray:
    return np.stack([x.ravel() for x in np.meshgrid(a, b, indexing='ij')], axis=1)


def gen_segments_all(df: pd.DataFrame) -> List[ndarray]:
    vert_i_by_layer = [g.index for _, g in df.groupby('layer')]
    return [gen_segments_layer(a, b) for a, b in zip(vert_i_by_layer, vert_i_by_layer[1:])]


def curvature_energy_matrix(pos: ndarray, s_ab: ndarray, s_bc: ndarray,
                            power: float = 3., cosine_threshold: float = 0.) -> coo_matrix:
    connected = coo_matrix(s_ab[:, 1, None] == s_bc[None, :, 0])
    s1 = s_ab[connected.row]
    s2 = s_bc[connected.col]
    w = curvature_energy_pairwise(
        pos[s1[:, 0]],
        pos[s1[:, 1]],
        pos[s2[:, 1]],
        power, cosine_threshold
    )
    m = coo_matrix((w, (connected.row, connected.col)), shape=(len(s_ab), len(s_bc)))
    m.eliminate_zeros()  # remove cosines below threshold completely
    return m


def curvature_energy_pairwise(a: ndarray, b: ndarray, c: ndarray,
                              power: float = 3., cosine_threshold: float = 0.) -> ndarray:
    d1 = b - a
    d2 = c - b
    r1 = np.linalg.norm(d1, axis=-1)
    r2 = np.linalg.norm(d2, axis=-1)
    rr = r1 * r2
    cosines = (d1 * d2).sum(axis=-1) / rr
    cosines[cosines < cosine_threshold] = 0
    return -0.5 * cosines ** power / rr


def curvature_energy(w: spmatrix, v1: ndarray, v2: ndarray) -> float:
    return w.dot(v2).dot(v1)


def curvature_energy_gradient(w: spmatrix, v1: ndarray, v2: ndarray) -> Tuple[ndarray, ndarray]:
    return w.dot(v2), w.transpose().dot(v1)


def count_segments(activation: Iterable[ndarray]) -> ndarray:
    return sum(v.sum() for v in activation)


def number_of_used_vertices_energy(vertex_count: int, segment_count: ndarray) -> ndarray:
    return 0.5 * (vertex_count - segment_count) ** 2


def number_of_used_vertices_energy_gradient(vertex_count: int, total_activation: float) -> float:
    return total_activation - vertex_count


def join_energy_matrix(segments: ndarray) -> csr_matrix:
    joins = (segments[:, None, 1] == segments[None, :, 1])
    np.fill_diagonal(joins, 0)
    return csr_matrix(joins)


def fork_energy_matrix(segments: ndarray) -> csr_matrix:
    forks = (segments[:, None, 0] == segments[None, :, 0])
    np.fill_diagonal(forks, 0)
    return csr_matrix(forks)


def layer_energy(matrix: Union[ndarray, spmatrix], activation: ndarray) -> float:
    return 0.5 * (matrix.dot(activation).dot(activation))


def layer_energy_gradient(matrix: Union[ndarray, spmatrix], activation: ndarray) -> ndarray:
    return matrix.dot(activation)


def energy(*args, **kwargs):
    ee = energies(*args, **kwargs)

    def _energy(activation):
        return sum(ee(activation))

    return _energy


def energy_gradient(*args, **kwargs):
    egs = energy_gradients(*args, **kwargs)

    def _energy_gradient(activation):
        ecg, eng, efg = egs(activation)
        return [ecg[i] + eng[i] + efg[i] for i in range(len(activation))]

    return _energy_gradient


def energies(pos: ndarray, segments: Iterable[ndarray], alpha: float = 1., beta: float = 1.,
             curvature_cosine_power: float = 3, cosine_threshold: float = 0.):
    seg_layers = segments, islice(segments, 1, None)
    curvature_matrices = [
        curvature_energy_matrix(pos, s_ab, s_bc,
                                power=curvature_cosine_power, cosine_threshold=cosine_threshold)
        for s_ab, s_bc in zip(*seg_layers)]
    seg = np.concatenate(segments) if segments else np.zeros(1)
    ls = len(seg)
    if curvature_matrices:
        curvature_matrix = sparse.block_diag(curvature_matrices, format='csr')
        w, h = curvature_matrix.shape
        left_margin = ls - w
        bottom_margin = ls - h
        curvature_matrix = sparse.hstack([
                csr_matrix(np.zeros((h, left_margin))),
                curvature_matrix
            ], 'csr')
        curvature_matrix = sparse.vstack([
            curvature_matrix,
            csr_matrix(np.zeros((bottom_margin, ls))),
        ], 'csr')
    else:
        curvature_matrix = csr_matrix(np.zeros((ls, ls)))
    n = len(pos)
    crossing_matrix = sparse.block_diag(
        [fork_energy_matrix(s).tocsr() + join_energy_matrix(s).tocsr() for s in segments],
        format="csr") if segments else 0

    def inner(activation):
        v = np.concatenate(activation) if activation else 0
        ec = curvature_energy(curvature_matrix, v, v)
        ef = alpha * layer_energy(crossing_matrix, np.concatenate(activation)) if activation else 0
        en = beta * number_of_used_vertices_energy(n, count_segments(activation))
        return ec, en, ef

    return inner


def energy_gradients(pos: ndarray, segments: Iterable[ndarray], alpha: float = 1., beta: float = 1.,
                     curvature_cosine_power: float = 3, cosine_threshold: float = 0.,
                     drop_gradients_on_self: bool = True):
    seg_layers = segments, islice(segments, 1, None)
    curvature_matrices = [
        curvature_energy_matrix(pos, s_ab, s_bc,
                                power=curvature_cosine_power, cosine_threshold=cosine_threshold)
        for s_ab, s_bc in zip(*seg_layers)]
    n = len(pos)
    crossing_matrices = [fork_energy_matrix(s) + join_energy_matrix(s) for s in segments]

    def _energy_gradients(activation):
        ec_g1g2 = [curvature_energy_gradient(w, v1, v2) for w, v1, v2 in
                   zip(curvature_matrices, activation, islice(activation, 1, None))]
        ecg = [np.zeros_like(a) for a in activation]
        for i in range(len(ecg)):
            if i < len(ec_g1g2):
                ecg[i] += ec_g1g2[i][0]
            if i > 0:
                ecg[i] += ec_g1g2[i - 1][1]

        efg = [alpha * layer_energy_gradient(m, v) for v, m in zip(activation, crossing_matrices)]
        total_act = sum(v.sum() for v in activation)
        eng = [beta * np.full_like(v, number_of_used_vertices_energy_gradient(n, total_act)) for v in activation]
        if drop_gradients_on_self:
            for e, a in zip(eng, activation):
                e -= a
        return ecg, eng, efg

    return _energy_gradients
