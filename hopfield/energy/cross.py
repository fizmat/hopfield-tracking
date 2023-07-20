import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import cartesian


def group_cross(hit_ids: ndarray) -> csr_matrix:
    df = pd.DataFrame({'hit_id': hit_ids})
    ii = []
    jj = []
    for end, group in df.groupby('hit_id'):
        ij = cartesian([group.index, group.index])
        ii.append(ij[:, 0])
        jj.append(ij[:, 1])
    ii = np.concatenate(ii) if ii else np.empty(0, int)
    jj = np.concatenate(jj) if jj else np.empty(0, int)
    non_diagonal = ii != jj
    ii = ii[non_diagonal]
    jj = jj[non_diagonal]
    n = len(hit_ids)
    return csr_matrix((np.ones(len(ii), dtype=int), (ii, jj)), shape=(n, n))


def segment_forks(seg: ndarray) -> csr_matrix:
    return group_cross(seg[:, 0])


def segment_joins(seg: ndarray) -> csr_matrix:
    return group_cross(seg[:, 1])


def cross_energy_matrix(seg: ndarray) -> csr_matrix:
    return segment_forks(seg) + segment_joins(seg)
