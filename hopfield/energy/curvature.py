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


def curvature_energy(cosines: np.ndarray, denominator: np.ndarray,
                     alpha: float, gamma: float,
                     cosine_power: float = 3., cosine_threshold: float = 0.,
                     distance_prod_power_in_denominator: float = 1.,
                     cosine_min_allowed: float = -2.) -> ndarray:
    curve = cosines.copy()
    curve[cosines < cosine_threshold] = 0
    curve = curve ** cosine_power / denominator ** distance_prod_power_in_denominator
    kink = (cosines < cosine_min_allowed).astype(int)
    return alpha * kink - gamma * curve


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


def curvature_energy_matrix(pos: ndarray, seg: ndarray, pairs: ndarray,
                            alpha: float, gamma: float,
                            curvature_cosine_power: float = 3.,
                            cosine_threshold: float = 0.,
                            do_sum_r: bool = True,
                            distance_prod_power_in_denominator: float = 1.,
                            cosine_min_allowed: float = -2.) -> csr_matrix:
    if len(pairs) == 0:
        return csr_matrix(np.empty((len(seg), len(seg)), dtype=float))
    s1, s2 = seg[pairs.T]
    a, b, c = pos[s1[:, 0]], pos[s1[:, 1]], pos[s2[:, 1]]
    cosines, r1, r2 = _curvature_pairwise(a, b, c)
    denominator = r1 + r2 if do_sum_r else r1 * r2
    w = curvature_energy(cosines, denominator, alpha, gamma, cosine_power=curvature_cosine_power,
                         cosine_threshold=cosine_threshold,
                         distance_prod_power_in_denominator=distance_prod_power_in_denominator,
                         cosine_min_allowed=cosine_min_allowed)
    m = coo_matrix((w, pairs.T), shape=(len(seg), len(seg)))
    m.eliminate_zeros()
    return (m + m.transpose()).tocsr()
