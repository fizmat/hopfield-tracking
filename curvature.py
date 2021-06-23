from typing import Tuple, List

import numpy as np
from numpy import ndarray
from scipy.sparse import coo_matrix, spmatrix, csr_matrix


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


def segment_find_next(seg: ndarray, i: int) -> ndarray:
    is_adjacent = seg[:, 0] == seg[i, 1]
    jj = np.arange(len(seg))[is_adjacent]
    segments = np.stack(np.broadcast_arrays(i, jj), axis=1)
    return segments


def segment_adjacent_pairs(seg: ndarray) -> ndarray:
    if len(seg) == 0:
        return np.empty((0, 2), dtype=int)
    return np.concatenate([segment_find_next(seg, s) for s in range(len(seg))])


def curvature_energy_matrix(pos: ndarray, segments: List[ndarray],
                            curvature_cosine_power: float = 3,
                            cosine_threshold: float = 0.) -> csr_matrix:
    if len(segments) == 0:
        return csr_matrix(np.empty((0, 0)))
    seg = np.concatenate(segments)
    pairs = segment_adjacent_pairs(seg).transpose()
    s1, s2 = seg[pairs]
    a, b, c = pos[s1[:, 0]], pos[s1[:, 1]], pos[s2[:, 1]]
    w = curvature_energy_pairwise(a, b, c,power=curvature_cosine_power,
                                  cosine_threshold=cosine_threshold)
    return coo_matrix((w, pairs), shape=(len(seg), len(seg))).tocsr()


def curvature_energy(w: spmatrix, v1: ndarray, v2: ndarray) -> float:
    return w.dot(v2).dot(v1)


def curvature_energy_gradient(w: spmatrix, v1: ndarray, v2: ndarray) -> Tuple[ndarray, ndarray]:
    return w.dot(v2), w.transpose().dot(v1)
