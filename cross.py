from typing import Union

import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix, spmatrix


def join_energy_matrix(segments: ndarray) -> csr_matrix:
    joins = (segments[:, None, 1] == segments[None, :, 1])
    np.fill_diagonal(joins, 0)
    return csr_matrix(joins)


def fork_energy_matrix(segments: ndarray) -> csr_matrix:
    forks = (segments[:, None, 0] == segments[None, :, 0])
    np.fill_diagonal(forks, 0)
    return csr_matrix(forks)


def layer_energy(matrix: Union[ndarray, spmatrix], activation: ndarray) -> float:
    return 0.5 * (matrix.dot(activation).dot(activation))


def layer_energy_gradient(matrix: Union[ndarray, spmatrix], activation: ndarray) -> ndarray:
    return matrix.dot(activation)