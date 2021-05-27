from itertools import islice
from typing import Iterable

import torch
from torch import Tensor, cdist, einsum, tensordot
from torch.types import Number


def curvature_energy_matrix(a: Tensor, b: Tensor, c: Tensor,
                            power: float = 3., cosine_threshold: float = 0.):
    r1 = cdist(a, b)  # ij
    r2 = cdist(b, c)  # jk
    d1 = b[None, :, :] - a[:, None, :]  # ijc
    d2 = c[None, :, :] - b[:, None, :]  # jkc
    rr = r1[:, :, None] * r2[None, :, :]  # ijk
    cosines = einsum('ijc,jkc->ijk', d1, d2) / rr  # ijk
    cosines[cosines < cosine_threshold] = 0
    return -0.5 * torch.pow(cosines, power) / rr  # ijk


def curvature_energy(w_ijk: Tensor, v_ij: Tensor, v_jk: Tensor):
    return (w_ijk * v_ij[:, :, None] * v_jk[None, :, :]).sum()


def count_vertices(pos: Iterable[Tensor]) -> int:
    return sum(map(len, pos))


def count_segments(activation: Iterable[Tensor]) -> Tensor:
    return sum(v.sum() for v in activation)


def number_of_used_vertices_energy(vertex_count: int, segment_count: Tensor) -> Tensor:
    return 0.5 * (vertex_count - segment_count) ** 2


def join_energy(activation: Tensor) -> Tensor:
    v = activation
    return 0.5 * (tensordot(v, v, [[1], [1]]).sum() - tensordot(v, v, 2))


def fork_energy(activation: Tensor) -> Tensor:
    v = activation
    return 0.5 * (tensordot(v, v, [[0], [0]]).sum() - tensordot(v, v, 2))


def track_crossing_energy(v):
    return fork_energy(v) + join_energy(v)


def energy(*args, **kwargs):
    ee = energies(*args, **kwargs)

    def inner(activation):
        return sum(ee(activation))

    return inner


def energies(pos: Iterable[Tensor], alpha: Number = 1., beta: Number = 1.,
             curvature_cosine_power: Number = 3, cosine_threshold: float = 0.):
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
