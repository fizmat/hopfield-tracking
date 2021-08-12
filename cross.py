import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix, coo_matrix

from curvature import segment_adjacent_pairs


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


def segment_kinks_pairwise(a: ndarray, b: ndarray, c: ndarray, cosine_min_allowed: float = 0) -> ndarray:
    d1 = b - a
    d2 = c - b
    r1 = np.linalg.norm(d1, axis=-1)
    r2 = np.linalg.norm(d2, axis=-1)
    rr = r1 * r2
    cosines = (d1 * d2).sum(axis=-1) / rr
    return (cosines < cosine_min_allowed).astype(int)


def segment_kinks(seg: ndarray, pos: ndarray, cosine_min_allowed: float = 0, pairs: ndarray = None) -> csr_matrix:
    if pairs is None:
        pairs = segment_adjacent_pairs(seg)
    if len(pairs) == 0:
        return csr_matrix(np.empty((len(seg), len(seg)), dtype=int))
    pairs = pairs.transpose()
    s1, s2 = seg[pairs]
    a, b, c = pos[s1[:, 0]], pos[s1[:, 1]], pos[s2[:, 1]]
    w = segment_kinks_pairwise(a, b, c, cosine_min_allowed)
    m = coo_matrix((w, pairs), shape=(len(seg), len(seg)))
    return (m + m.transpose()).tocsr()


def cross_energy_matrix(seg: ndarray, pos: ndarray = None, cosine_min_allowed: float = None,
                        pairs: ndarray = None) -> csr_matrix:
    m = segment_forks(seg) + segment_joins(seg)
    if cosine_min_allowed is not None:
        m += segment_kinks(seg, pos, cosine_min_allowed, pairs)
    return m
