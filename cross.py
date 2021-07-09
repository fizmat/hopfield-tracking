import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix


def segment_forks(seg: ndarray) -> csr_matrix:
    n = len(seg)
    if n == 0:
        return csr_matrix(np.empty((0, 0), dtype=int))
    is_fork = np.logical_and(
        np.equal(seg[:, np.newaxis, 0], seg[np.newaxis, :, 0]),
        np.logical_not(np.eye(n, dtype=bool))
    )
    return csr_matrix(is_fork).astype(int)


def segment_joins(seg: ndarray) -> csr_matrix:
    n = len(seg)
    if n == 0:
        return csr_matrix(np.empty((0, 0), dtype=int))
    is_fork = np.logical_and(
        np.equal(seg[:, np.newaxis, 1], seg[np.newaxis, :, 1]),
        np.logical_not(np.eye(n, dtype=bool))
    )
    return csr_matrix(is_fork).astype(int)


def cross_energy_matrix(seg: ndarray) -> csr_matrix:
    return segment_forks(seg) + segment_joins(seg)
