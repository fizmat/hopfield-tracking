import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from tracking.segment import _gen_segments_layer, gen_segments_all, gen_seg_track_sequential

_hits = pd.DataFrame({'track': [1, -1, 3, 5, -1, 3, 5, 1, 3],
                      'layer': [0, 0, 0, 1, 1, 1, 2, 2, 2]})


def test_gen_segments_layer():
    a = np.arange(2)
    b = np.arange(4)
    assert_array_equal(_gen_segments_layer(a, b), [[0, 0], [0, 1], [0, 2], [0, 3],
                                                   [1, 0], [1, 1], [1, 2], [1, 3]])


def test_gen_segment_all():
    df = pd.DataFrame({'x': 0, 'y': 0, 'z': 0, 'layer': [0, 0, 1, 1, 1, 2], 'track': 0})
    seg = gen_segments_all(df)
    assert_array_equal(seg, [[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 5], [3, 5], [4, 5]])


def test_gen_seg_track_sequential():
    assert_array_equal(list(gen_seg_track_sequential(_hits)), [(0, 7), (2, 5), (5, 8), (3, 6)])
