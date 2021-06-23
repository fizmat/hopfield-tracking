from typing import Union, List

import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix, spmatrix, coo_matrix


def segment_one_forks(seg: ndarray, i: int) -> ndarray:
    is_fork = np.equal(seg[:, 0], seg[i, 0])
    is_fork[i] = False
    jj = np.arange(len(seg))[is_fork]
    segments = np.stack(np.broadcast_arrays(i, jj), axis=1)
    return segments


def segment_forks(seg: ndarray) -> ndarray:
    if len(seg) == 0:
        return np.empty((0, 2), dtype=int)
    return np.concatenate([segment_one_forks(seg, s) for s in range(len(seg))])


def segment_one_joins(seg: ndarray, i: int) -> ndarray:
    is_join = np.equal(seg[:, 1], seg[i, 1])
    is_join[i] = False
    jj = np.arange(len(seg))[is_join]
    segments = np.stack(np.broadcast_arrays(i, jj), axis=1)
    return segments


def segment_joins(seg: ndarray) -> ndarray:
    if len(seg) == 0:
        return np.empty((0, 2), dtype=int)
    return np.concatenate([segment_one_joins(seg, s) for s in range(len(seg))])


def cross_energy_matrix(segments: List[ndarray]) -> csr_matrix:
    if len(segments) < 1:
        return csr_matrix(np.zeros((0, 0)))
    seg = np.concatenate(segments)
    crosses = np.concatenate([segment_forks(seg), segment_joins(seg)])
    return coo_matrix((np.ones(len(crosses)), crosses.transpose()), shape=(len(seg), len(seg))).tocsr()


def cross_energy(matrix: Union[ndarray, spmatrix], activation: ndarray) -> float:
    return 0.5 * (matrix.dot(activation).dot(activation))


def cross_energy_gradient(matrix: Union[ndarray, spmatrix], activation: ndarray) -> ndarray:
    return matrix.dot(activation)
