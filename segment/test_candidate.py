import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from segment.candidate import gen_seg_all, _gen_seg_one_layer, gen_seg_layered

_hits = pd.DataFrame({'layer': [0, 0, 0, 1, 1, 1, 2, 2, 2],
                      'hit_id': [1, 10, 3, 9, 4, 8, 5, 7, 6]}).set_index('hit_id')


def test_gen_seg_all():
    assert_array_equal(gen_seg_all(_hits), [(1, 10), (3, 10), (9, 10), (4, 10), (8, 10), (5, 10), (7, 10), (6, 10),
                                            (1, 3), (1, 9), (3, 9), (4, 9), (8, 9), (5, 9), (7, 9), (6, 9),
                                            (1, 4), (3, 4), (1, 8), (3, 8), (4, 8), (5, 8), (7, 8), (6, 8),
                                            (1, 5), (3, 5), (4, 5), (1, 7), (3, 7), (4, 7), (5, 7), (6, 7),
                                            (1, 6), (3, 6), (4, 6), (5, 6)])


def test__gen_seg_one_layer():
    a = np.arange(2)
    b = np.arange(4)
    assert_array_equal(_gen_seg_one_layer(a, b), [[0, 0], [0, 1], [0, 2], [0, 3],
                                                  [1, 0], [1, 1], [1, 2], [1, 3]])


def test_gen_seg_layered():
    df = pd.DataFrame({'x': 0, 'y': 0, 'z': 0, 'layer': [0, 0, 1, 1, 1, 2], 'track': 0})
    seg = gen_seg_layered(df)
    assert_array_equal(seg, [[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 5], [3, 5], [4, 5]])
