from typing import Union, List

import numpy as np
from numpy import ndarray
from scipy import sparse
from scipy.sparse import csr_matrix, spmatrix


def fork_layer_matrix(segments: ndarray) -> csr_matrix:
    if len(segments) < 1:
        return csr_matrix(np.zeros((0, 0)))
    forks = (segments[:, None, 0] == segments[None, :, 0]).astype(float)
    np.fill_diagonal(forks, 0)
    return csr_matrix(forks)


def join_layer_matrix(segments: ndarray) -> csr_matrix:
    if len(segments) < 1:
        return csr_matrix(np.zeros((0, 0)))
    joins = (segments[:, None, 1] == segments[None, :, 1]).astype(float)
    np.fill_diagonal(joins, 0)
    return csr_matrix(joins)


def cross_energy_matrix(segments: List[ndarray]) -> csr_matrix:
    if len(segments) < 1:
        return csr_matrix(np.zeros((0, 0)))
    return sparse.block_diag(
        [fork_layer_matrix(s).tocsr() + join_layer_matrix(s).tocsr() for s in segments],
        format="csr")


def cross_energy(matrix: Union[ndarray, spmatrix], activation: ndarray) -> float:
    return 0.5 * (matrix.dot(activation).dot(activation))


def cross_energy_gradient(matrix: Union[ndarray, spmatrix], activation: ndarray) -> ndarray:
    return matrix.dot(activation)
