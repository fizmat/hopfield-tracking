import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix, coo_matrix


def segment_forks(seg: ndarray) -> csr_matrix:
    n = len(seg)
    if n == 0:
        return csr_matrix(np.empty((0, 0), dtype=int))
    starts = seg[:, 0]
    ii = []
    jj = []
    for i in range(n):
        is_fork = starts == starts[i]
        is_fork[i] = False
        jj.append(is_fork.nonzero()[0])
        ii.append([i] * len(jj[-1]))
    jj = [j for aj in jj for j in aj]
    ii = [i for ai in ii for i in ai]
    return coo_matrix((np.ones(len(ii), dtype=int), (ii, jj)), shape=(n, n)).tocsr()


def segment_joins(seg: ndarray) -> csr_matrix:
    n = len(seg)
    if n == 0:
        return csr_matrix(np.empty((0, 0), dtype=int))
    ends = seg[:, 1]
    ii = []
    jj = []
    for i in range(n):
        is_join = ends == ends[i]
        is_join[i] = False
        jj.append(is_join.nonzero()[0])
        ii.append([i] * len(jj[-1]))
    jj = [j for aj in jj for j in aj]
    ii = [i for ai in ii for i in ai]
    return coo_matrix((np.ones(len(ii), dtype=int), (ii, jj)), shape=(n, n)).tocsr()


def cross_energy_matrix(seg: ndarray) -> csr_matrix:
    return segment_forks(seg) + segment_joins(seg)
