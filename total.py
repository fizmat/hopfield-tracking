from typing import Tuple

import numpy as np
from numpy import ndarray


def total_activation_matrix_(vertex_count: int, segment_count: int, drop_gradients_on_self: bool = True) \
        -> Tuple[ndarray, ndarray, float]:
    a = np.ones((segment_count, segment_count))
    if drop_gradients_on_self:
        a -= np.eye(segment_count)
    b = - 2 * np.full(segment_count, vertex_count)
    c = vertex_count ** 2
    return a, b, c


def total_activation_matrix(pos: ndarray, seg: ndarray, drop_gradients_on_self: bool = True) \
        -> Tuple[ndarray, ndarray, float]:
    return total_activation_matrix_(len(pos), len(seg), drop_gradients_on_self)


def total_activation_energy(a: ndarray, b: ndarray, c: float, activation: ndarray) -> float:
    return a.dot(activation).dot(activation) + b.dot(activation) + c


def total_activation_energy_gradient(a: ndarray, b: ndarray, activation: ndarray) -> float:
    return 2 * a.dot(activation) + b
