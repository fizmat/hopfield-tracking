from typing import Union

from numpy import ndarray
from scipy.sparse import spmatrix


def energy(matrix: Union[spmatrix, ndarray], act: ndarray):
    return matrix.dot(act).dot(act)


def energy_gradient(matrix: Union[spmatrix, ndarray], act: ndarray):
    return 2 * matrix.dot(act)