from typing import Tuple, Iterable

import numpy as np
from numpy import ndarray


def total_activation_matrix_(vertex_count: int, segment_count: int) \
        -> Tuple[ndarray, ndarray, float]:
    a = np.full((segment_count, segment_count), 0.5)
    b = - np.full(segment_count, vertex_count)
    c = 0.5 * vertex_count ** 2
    return a, b, c


def total_activation_matrix(pos: ndarray, seg: Iterable[ndarray]) \
        -> Tuple[ndarray, ndarray, float]:
    return total_activation_matrix_(len(pos), sum(len(s) for s in seg))


def total_activation_energy(a: ndarray, b: ndarray, c: float, activation: ndarray) -> float:
    return a.dot(activation).dot(activation) + b.dot(activation) + c


def total_activation_energy_gradient(vertex_count: int, total_activation: float) -> float:
    return total_activation - vertex_count
