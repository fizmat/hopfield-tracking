from typing import Tuple

import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.utils.extmath import cartesian


def _curvature_pairwise(a: ndarray, b: ndarray, c: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
    d1 = b - a
    d2 = c - b
    r1 = np.linalg.norm(d1, axis=-1)
    r2 = np.linalg.norm(d2, axis=-1)
    rr = r1 * r2
    cosines = (d1 * d2).sum(axis=-1) / rr
    return cosines, r1, r2


def _curvature_energy_pairwise(cosines: np.ndarray, r1: np.ndarray, r2: np.ndarray,
                               cosine_power: float = 3., cosine_threshold: float = 0.,
                               distance_power: float = 1., sum_distances: bool = True) -> ndarray:
    curve = cosines.copy()
    curve[cosines < cosine_threshold] = 0
    denominator = r1 + r2 if sum_distances else r1 * r2
    return curve ** cosine_power / denominator ** distance_power


def _kink_energy_pairwise(cosines: np.ndarray, kink_threshold: float = 0.) -> ndarray:
    return (cosines < kink_threshold).astype(int)


def _find_indices_with_equal_values(a: ndarray, b: ndarray) -> Tuple[ndarray, ndarray]:
    a = pd.DataFrame({'hit_id': a}).groupby('hit_id').apply(lambda g: g.index)
    b = pd.DataFrame({'hit_id': b}).groupby('hit_id').apply(lambda g: g.index)
    df = pd.concat((a, b), axis=1, join='inner')
    ii = []
    jj = []
    for _, (a, b) in df.iterrows():
        ij = cartesian([a, b])
        ii.append(ij[:, 0])
        jj.append(ij[:, 1])
    ii = np.concatenate(ii) if ii else np.empty(0, int)
    jj = np.concatenate(jj) if jj else np.empty(0, int)
    return ii, jj


def find_consecutive_segments(seg: ndarray) -> ndarray:
    ii, jj = _find_indices_with_equal_values(seg[:, 1], seg[:, 0])
    return np.stack([ii, jj], axis=1)


def prep_curvature(pos: ndarray, seg: ndarray) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    pairs = find_consecutive_segments(seg)
    s1, s2 = seg[pairs.T]
    a, b, c = pos[s1[:, 0]], pos[s1[:, 1]], pos[s2[:, 1]]
    cosines, r1, r2 = _curvature_pairwise(a, b, c)
    return pairs, cosines, r1, r2


def curvature_energy_matrix(size: int, pairs: ndarray,
                            cosines: ndarray, r1: ndarray, r2: ndarray,
                            cosine_power: float = 3., cosine_threshold: float = 0.,
                            distance_power: float = 1., do_sum_distances: bool = True):
    v = _curvature_energy_pairwise(cosines, r1, r2, cosine_power, cosine_threshold, distance_power, do_sum_distances)
    m = coo_matrix((v, pairs.T), shape=(size, size))
    m.eliminate_zeros()
    m = m.tocsr()
    return m + m.transpose()


def kink_energy_matrix(size: int, pairs: ndarray, cosines: ndarray, kink_threshold: float = 0.) -> csr_matrix:
    v = _kink_energy_pairwise(cosines, kink_threshold)
    m = coo_matrix((v, pairs.T), shape=(size, size)).tocsr()
    return m + m.transpose()
