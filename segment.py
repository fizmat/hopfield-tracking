from itertools import islice
from typing import Iterable, List, Tuple

import numpy as np
from numpy import ndarray
from scipy.sparse import coo_matrix, spmatrix


def gen_segments_layer(a: ndarray, b: ndarray):
    return np.mgrid[0:len(a), 0:len(b)].reshape((2, -1))


def gen_segments_all(pos: Iterable[ndarray]):
    return [gen_segments_layer(a, b) for a, b in zip(pos, pos[1:])]


def curvature_energy_matrix(a: ndarray, b: ndarray, c: ndarray, s_ab: ndarray, s_bc: ndarray,
                            power: float = 3., cosine_threshold: float = 0.) -> coo_matrix:
    connected = coo_matrix(s_ab[1, :, None] == s_bc[None, 0, :])
    s1 = s_ab[:, connected.row]
    s2 = s_bc[:, connected.col]
    w = curvature_energy_pairwise(
        a[s1[0]],
        b[s1[1]],
        c[s2[1]],
        power, cosine_threshold
    )
    m = coo_matrix((w, (connected.row, connected.col)), shape=(s_ab.shape[-1], s_bc.shape[-1]))
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


def count_vertices(pos: Iterable[ndarray]) -> int:
    return sum(map(len, pos))


def count_segments(activation: Iterable[ndarray]) -> ndarray:
    return sum(v.sum() for v in activation)


def number_of_used_vertices_energy(vertex_count: int, segment_count: ndarray) -> ndarray:
    return 0.5 * (vertex_count - segment_count) ** 2


def number_of_used_vertices_energy_gradient(vertex_count: int, total_activation: float) -> float:
    return total_activation - vertex_count


def join_energy(activation: ndarray, segments: ndarray) -> float:
    joins = (segments[1, :, None] == segments[1, None, :])
    np.fill_diagonal(joins, 0)

    return 0.5 * (joins.dot(activation).dot(activation))


def join_energy_gradient(activation_layer: ndarray, segments: ndarray) -> ndarray:
    joins = (segments[1, :, None] == segments[1, None, :])
    np.fill_diagonal(joins, 0)
    return joins.dot(activation_layer)


def fork_energy(activation: ndarray, segments: ndarray) -> float:
    forks = (segments[0, :, None] == segments[0, None, :])
    np.fill_diagonal(forks, 0)
    return 0.5 * (forks.dot(activation).dot(activation))


def fork_energy_gradient(activation_layer: ndarray, segments: ndarray) -> ndarray:
    forks = (segments[0, :, None] == segments[0, None, :])
    np.fill_diagonal(forks, 0)
    return forks.dot(activation_layer)


def track_crossing_energy(activation: ndarray, segments: ndarray) -> float:
    return fork_energy(activation, segments) + join_energy(activation, segments)


def track_crossing_energy_gradient(activation: ndarray, segments: ndarray) -> ndarray:
    return fork_energy_gradient(activation, segments) + join_energy_gradient(activation, segments)


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


def energies(pos: Iterable[ndarray], segments: Iterable[ndarray], alpha: float = 1., beta: float = 1.,
             curvature_cosine_power: float = 3, cosine_threshold: float = 0.):
    pos_layers = pos, islice(pos, 1, None), islice(pos, 2, None)
    seg_layers = segments, islice(segments, 1, None)
    curvature_matrices = [
        curvature_energy_matrix(a, b, c, s_ab, s_bc,
                                power=curvature_cosine_power, cosine_threshold=cosine_threshold)
        for a, b, c, s_ab, s_bc in zip(*pos_layers, *seg_layers)]
    n = count_vertices(pos)

    def inner(activation):
        ec = sum(curvature_energy(w, v1, v2) for w, v1, v2 in
                 zip(curvature_matrices, activation, islice(activation, 1, None)))
        ef = alpha * sum(track_crossing_energy(v, s) for v, s in zip(activation, segments))
        en = beta * number_of_used_vertices_energy(n, count_segments(activation))
        return ec, en, ef

    return inner


def energy_gradients(pos: Iterable[ndarray], segments: Iterable[ndarray], alpha: float = 1., beta: float = 1.,
                     curvature_cosine_power: float = 3, cosine_threshold: float = 0.,
                     drop_gradients_on_self: bool = True):
    pos_layers = pos, islice(pos, 1, None), islice(pos, 2, None)
    seg_layers = segments, islice(segments, 1, None)
    curvature_matrices = [
        curvature_energy_matrix(a, b, c, s_ab, s_bc,
                                power=curvature_cosine_power, cosine_threshold=cosine_threshold)
        for a, b, c, s_ab, s_bc in zip(*pos_layers, *seg_layers)]
    n = count_vertices(pos)

    def _energy_gradients(activation):
        ec_g1g2 = [curvature_energy_gradient(w, v1, v2) for w, v1, v2 in
                   zip(curvature_matrices, activation, islice(activation, 1, None))]
        ecg = [np.zeros_like(a) for a in activation]
        for i in range(len(ecg)):
            if i < len(ec_g1g2):
                ecg[i] += ec_g1g2[i][0]
            if i > 0:
                ecg[i] += ec_g1g2[i - 1][1]

        efg = [alpha * track_crossing_energy_gradient(v, s) for v, s in zip(activation, segments)]
        total_act = sum(v.sum() for v in activation)
        eng = [beta * np.full_like(v, number_of_used_vertices_energy_gradient(n, total_act)) for v in activation]
        if drop_gradients_on_self:
            for e, a in zip(eng, activation):
                e -= a
        return ecg, eng, efg

    return _energy_gradients
