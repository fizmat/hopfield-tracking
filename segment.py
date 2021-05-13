from typing import Callable

import torch
from torch import Tensor, cdist, einsum, tensordot

M = 2


def curvature(a: Tensor, b: Tensor, c: Tensor) -> Callable[[Tensor, Tensor], Tensor]:
    r1 = cdist(a, b)  # ij
    r2 = cdist(b, c)  # jk
    d1 = b[None, :, :] - a[:, None, :]  # ijc
    d2 = c[None, :, :] - b[:, None, :]  # jkc
    rr = r1[:, :, None] * r2[None, :, :]

    def inner(v1: Tensor, v2: Tensor) -> Tensor:
        cosines = einsum('ijc,jkc->ijk', d1, d2) / rr
        return - 0.5 * (torch.pow(cosines, M) * v1[:, :, None] * v2[None, :, :] / rr).sum()

    return inner


beta = 3.


def T2(a, b, c):
    N = len(a) + len(b) + len(c)

    def inner(v1, v2):
        return beta * torch.square(0.5 * (v1.sum() + v2.sum() - N))

    return inner


alpha = 0.2


def T1(a, b, c):
    def inner(v1, v2):
        return alpha / 2 * (tensordot(v1, v1, [[0], [0]]).sum() - tensordot(v1, v1, 2) +
                            tensordot(v1, v1, [[1], [1]]).sum() - tensordot(v1, v1, 2) +
                            tensordot(v2, v2, [[0], [0]]).sum() - tensordot(v2, v2, 2) +
                            tensordot(v2, v2, [[1], [1]]).sum() - tensordot(v2, v2, 2))

    return inner


def energy(a, b, c):
    E1 = curvature(a, b, c)
    t1 = T1(a, b, c)
    t2 = T2(a, b, c)

    def inner(v1, v2):
        return E1(v1, v2) + t1(v1, v2) + t2(v1, v2)

    return inner
