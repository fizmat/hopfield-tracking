import numpy as np
from numpy import ndarray
from scipy.sparse import coo_matrix, spmatrix, csr_matrix


def curvature_energy_pairwise(a: ndarray, b: ndarray, c: ndarray,
                              cosine_power: float = 3., cosine_threshold: float = 0.,
                              distance_prod_power_in_denominator: float = 1.) -> ndarray:
    d1 = b - a
    d2 = c - b
    r1 = np.linalg.norm(d1, axis=-1)
    r2 = np.linalg.norm(d2, axis=-1)
    rr = r1 * r2
    cosines = (d1 * d2).sum(axis=-1) / rr
    cosines[cosines < cosine_threshold] = 0
    return -0.5 * cosines ** cosine_power / rr ** distance_prod_power_in_denominator


def segment_find_next(seg: ndarray, i: int) -> ndarray:
    is_adjacent = seg[:, 0] == seg[i, 1]
    jj = np.arange(len(seg))[is_adjacent]
    segments = np.stack(np.broadcast_arrays(i, jj), axis=1)
    return segments


def segment_adjacent_pairs(seg: ndarray) -> ndarray:
    if len(seg) == 0:
        return np.empty((0, 2), dtype=int)
    return np.concatenate([segment_find_next(seg, s) for s in range(len(seg))])


def curvature_energy_matrix(pos: ndarray, seg: ndarray,
                            curvature_cosine_power: float = 3.,
                            cosine_threshold: float = 0., distance_prod_power_in_denominator: float = 1.) -> csr_matrix:
    pairs = segment_adjacent_pairs(seg).transpose()
    s1, s2 = seg[pairs]
    a, b, c = pos[s1[:, 0]], pos[s1[:, 1]], pos[s2[:, 1]]
    w = curvature_energy_pairwise(a, b, c, cosine_power=curvature_cosine_power,
                                  cosine_threshold=cosine_threshold,
                                  distance_prod_power_in_denominator=distance_prod_power_in_denominator)
    m = coo_matrix((w, pairs), shape=(len(seg), len(seg)))
    return (m + m.transpose()).tocsr()


def curvature_energy(w: spmatrix, act: ndarray) -> float:
    return 0.5 * w.dot(act).dot(act)


def curvature_energy_gradient(w: spmatrix, act: ndarray) -> ndarray:
    return w.dot(act)
