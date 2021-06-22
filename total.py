from typing import Tuple, List

import numpy as np
from numpy import ndarray


def total_activation_matrix_(vertex_count: int, segment_count: int, drop_gradients_on_self: bool = True) \
        -> Tuple[ndarray, ndarray, float]:
    a = np.full((segment_count, segment_count), 0.5)
    if drop_gradients_on_self:
        a -= 0.5 * np.eye(segment_count)
    b = - np.full(segment_count, vertex_count)
    c = 0.5 * vertex_count ** 2
    return a, b, c


def total_activation_matrix(pos: ndarray, seg: List[ndarray], drop_gradients_on_self: bool = True) \
        -> Tuple[ndarray, ndarray, float]:
    return total_activation_matrix_(len(pos), sum(len(s) for s in seg), drop_gradients_on_self)


def total_activation_energy(a: ndarray, b: ndarray, c: float, activation: ndarray) -> float:
    return a.dot(activation).dot(activation) + b.dot(activation) + c


def total_activation_energy_gradient(a: ndarray, b: ndarray, activation: ndarray) -> float:
    return 2 * a.dot(activation) + b
