from typing import Callable

import torch
from torch import Tensor, cdist, einsum, tensordot


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


def number_of_used_vertices_energy(a, b, c):
    N = len(a) + len(b) + len(c)

    def inner(v1, v2):
        return torch.square(0.5 * (v1.sum() + v2.sum() - N))

    return inner


def number_of_forks_energy(v):
    join = 0.5 * (tensordot(v, v, [[1], [1]]).sum() - tensordot(v, v, 2))
    fork = 0.5 * (tensordot(v, v, [[0], [0]]).sum() - tensordot(v, v, 2))
    return fork + join


def energy(a, b, c, alpha=1., beta=1., curvature_cosine_power=3):
    E1 = curvature_energy(a, b, c, power=curvature_cosine_power)
    t1 = number_of_used_vertices_energy(a, b, c)

    def inner(v1, v2):
        return E1(v1, v2) + alpha * t1(v1, v2) + beta * (number_of_forks_energy(v1)+number_of_forks_energy(v2))

    return inner
