from typing import Tuple

import numpy as np
from numpy import ndarray
from scipy.sparse import coo_matrix, spmatrix


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


def curvature_energy(w: spmatrix, v1: ndarray, v2: ndarray) -> float:
    return w.dot(v2).dot(v1)


def curvature_energy_gradient(w: spmatrix, v1: ndarray, v2: ndarray) -> Tuple[ndarray, ndarray]:
    return w.dot(v2), w.transpose().dot(v1)
