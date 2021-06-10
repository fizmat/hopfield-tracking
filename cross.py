from typing import Union, Iterable

import numpy as np
from numpy import ndarray
from scipy import sparse
from scipy.sparse import csr_matrix, spmatrix


def join_layer_matrix(segments: ndarray) -> csr_matrix:
    joins = (segments[:, None, 1] == segments[None, :, 1])
    np.fill_diagonal(joins, 0)
    return csr_matrix(joins)


def fork_layer_matrix(segments: ndarray) -> csr_matrix:
    forks = (segments[:, None, 0] == segments[None, :, 0])
    np.fill_diagonal(forks, 0)
    return csr_matrix(forks)


def cross_energy_matrix(segments: Iterable[ndarray]) -> csr_matrix:
    return sparse.block_diag(
        [fork_layer_matrix(s).tocsr() + join_layer_matrix(s).tocsr() for s in segments],
        format="csr") if segments else csr_matrix(np.empty(0))


def cross_energy(matrix: Union[ndarray, spmatrix], activation: ndarray) -> float:
    return 0.5 * (matrix.dot(activation).dot(activation))


def cross_energy_gradient(matrix: Union[ndarray, spmatrix], activation: ndarray) -> ndarray:
    return matrix.dot(activation)
