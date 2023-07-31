import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import cartesian


def _find_index_pairs_with_equal_values(hit_ids: ndarray) -> csr_matrix:
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


def find_forking_segments(seg: ndarray) -> csr_matrix:
    return _find_index_pairs_with_equal_values(seg[:, 0])


def find_joining_segments(seg: ndarray) -> csr_matrix:
    return _find_index_pairs_with_equal_values(seg[:, 1])


def cross_energy_matrix(seg: ndarray) -> csr_matrix:
    return find_forking_segments(seg) + find_joining_segments(seg)
