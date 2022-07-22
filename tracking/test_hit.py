import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal, assert_allclose

from tracking.hit import cylindric_coordinates, add_cylindric_coordinates


def test_cylindric_coordinates():
    r, phi = cylindric_coordinates(np.array([0, 1, 0, -1, 1]), np.array([0, 0, 1, 1, -1]))
    assert_allclose(r, [0, 1, 1, np.sqrt(2), np.sqrt(2)])
    assert_allclose(phi, [0, 0, np.pi / 2, np.pi * 3 / 4, - np.pi / 4])


def test_add_cynlindric_coordinates():
    hits = pd.DataFrame({'x': np.array([0, 1, 0, -1, 1]),
                         'y': np.array([0, 0, 1, 1, -1])})
    add_cylindric_coordinates(hits)
    assert_allclose(hits.r, [0, 1, 1, np.sqrt(2), np.sqrt(2)])
    assert_allclose(hits.phi, [0, 0, np.pi / 2, np.pi * 3 / 4, - np.pi / 4])

