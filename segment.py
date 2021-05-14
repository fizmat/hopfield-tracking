from itertools import islice
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


def energy(pos: Iterable[Tensor], alpha: Number = 1., beta: Number = 1., curvature_cosine_power: Number = 3):
    layers = pos, islice(pos, 1, None), islice(pos, 2, None)
    curves = [curvature_energy(a, b, c, power=curvature_cosine_power) for a, b, c in zip(*layers)]
    n = count_vertices(pos)

    def inner(activation):
        result = sum(curve(v1, v2) for curve, v1, v2 in zip(curves, activation, islice(activation, 1, None)))
        result += alpha * number_of_used_vertices_energy(n, count_segments(activation))
        result += beta * sum(track_crossing_energy(v) for v in activation)
        return result

    return inner
