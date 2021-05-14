from typing import Callable, Iterable

import torch
from torch import Tensor, cdist, einsum, tensordot
from torch.types import Number


def curvature_energy(a: Tensor, b: Tensor, c: Tensor, power: float = 3) -> Callable[[Tensor, Tensor], Tensor]:
    r1 = cdist(a, b)  # ij
    r2 = cdist(b, c)  # jk
    d1 = b[None, :, :] - a[:, None, :]  # ijc
    d2 = c[None, :, :] - b[:, None, :]  # jkc
    rr = r1[:, :, None] * r2[None, :, :]

    def inner(v1: Tensor, v2: Tensor) -> Tensor:
        cosines = einsum('ijc,jkc->ijk', d1, d2) / rr
        return - 0.5 * (torch.pow(cosines, power) * v1[:, :, None] * v2[None, :, :] / rr).sum()

    return inner


def count_vertices(pos: Iterable[Tensor]) -> int:
    return sum(map(len, pos))


def count_segments(activation: Iterable[Tensor]) -> Number:
    return sum(v.sum() for v in activation)


def number_of_used_vertices_energy(vertex_count, segment_count):
    return 0.5 * (vertex_count - segment_count) ** 2


def join_energy(activation: Tensor) -> Number:
    v = activation
    return 0.5 * (tensordot(v, v, [[1], [1]]).sum() - tensordot(v, v, 2))


def fork_energy(activation: Tensor) -> Number:
    v = activation
    return 0.5 * (tensordot(v, v, [[0], [0]]).sum() - tensordot(v, v, 2))


def track_crossing_energy(v):
    return fork_energy(v) + join_energy(v)


def energy(a, b, c, alpha=1., beta=1., curvature_cosine_power=3):
    E1 = curvature_energy(a, b, c, power=curvature_cosine_power)
    n = count_vertices((a, b, c))

    def inner(v1, v2):
        return E1(v1, v2) + \
               alpha * number_of_used_vertices_energy(n, count_segments((v1, v2))) + \
               beta * (track_crossing_energy(v1) + track_crossing_energy(v2))

    return inner
