from itertools import islice
from typing import Iterable

import numpy as np
from numpy import ndarray


def curvature_energy_matrix(a: ndarray, b: ndarray, c: ndarray,
                            power: float = 3., cosine_threshold: float = 0.) -> ndarray:
    return curvature_energy_pairwise(
        a[:, None, None, :],
        b[None, :, None, :],
        c[None, None, :, :],
        power, cosine_threshold
    )


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


def curvature_energy(w_ijk: ndarray, v_ij: ndarray, v_jk: ndarray):
    return (w_ijk * v_ij[:, :, None] * v_jk[None, :, :]).sum()


def curvature_energy_gradient(w_ijk, v_ij, v_jk):
    return (w_ijk * v_jk[None, :, :]).sum(axis=2), (w_ijk * v_ij[:, :, None]).sum(axis=0)


def count_vertices(pos: Iterable[ndarray]) -> int:
    return sum(map(len, pos))


def count_segments(activation: Iterable[ndarray]) -> ndarray:
    return sum(v.sum() for v in activation)


def number_of_used_vertices_energy(vertex_count: int, segment_count: ndarray) -> ndarray:
    return 0.5 * (vertex_count - segment_count) ** 2


def number_of_used_vertices_energy_gradient(vertex_count: int, total_activation: float) -> float:
    return total_activation - vertex_count


def join_energy(activation: ndarray) -> ndarray:
    v = activation
    return 0.5 * (np.tensordot(v, v, ([1], [1])).sum() - np.tensordot(v, v, 2))


def join_energy_gradient(activation_layer: ndarray) -> ndarray:
    return activation_layer.sum(axis=0, keepdims=True) - activation_layer


def fork_energy(activation: ndarray) -> ndarray:
    v = activation
    return 0.5 * (np.tensordot(v, v, ([0], [0])).sum() - np.tensordot(v, v, 2))


def fork_energy_gradient(activation_layer: ndarray) -> ndarray:
    return activation_layer.sum(axis=1, keepdims=True) - activation_layer


def track_crossing_energy(v):
    return fork_energy(v) + join_energy(v)


def track_crossing_energy_gradient(v: ndarray) -> ndarray:
    return fork_energy_gradient(v) + join_energy_gradient(v)


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


def energies(pos: Iterable[ndarray], alpha: float = 1., beta: float = 1.,
             curvature_cosine_power: float = 3, cosine_threshold: float = 0.):
    layers = pos, islice(pos, 1, None), islice(pos, 2, None)
    curvature_matrices = [
        curvature_energy_matrix(a, b, c, power=curvature_cosine_power, cosine_threshold=cosine_threshold)
        for a, b, c in zip(*layers)]
    n = count_vertices(pos)

    def inner(activation):
        ec = sum(curvature_energy(w, v1, v2) for w, v1, v2 in
                 zip(curvature_matrices, activation, islice(activation, 1, None)))
        ef = alpha * sum(track_crossing_energy(v) for v in activation)
        en = beta * number_of_used_vertices_energy(n, count_segments(activation))
        return ec, en, ef

    return inner


def energy_gradients(pos: Iterable[ndarray], alpha: float = 1., beta: float = 1.,
                     curvature_cosine_power: float = 3, cosine_threshold: float = 0.,
                     drop_gradients_on_self: bool = True):
    layers = pos, islice(pos, 1, None), islice(pos, 2, None)
    curvature_matrices = [
        curvature_energy_matrix(a, b, c, power=curvature_cosine_power, cosine_threshold=cosine_threshold)
        for a, b, c in zip(*layers)]
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

        efg = [alpha * track_crossing_energy_gradient(v) for v in activation]
        total_act = sum(v.sum() for v in activation)
        eng = [beta * np.full_like(v, number_of_used_vertices_energy_gradient(n, total_act)) for v in activation]
        if drop_gradients_on_self:
            for e, a in zip(eng, activation):
                e -= a
        return ecg, eng, efg

    return _energy_gradients
