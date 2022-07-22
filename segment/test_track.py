from numpy.testing import assert_array_equal
import pandas as pd

from segment.track import gen_seg_track_sequential, gen_seg_track_layered

_hits = pd.DataFrame({'track': [1, -1, 3, 5, -1, 3, 5, 1, 3],
                      'layer': [0, 0, 0, 1, 1, 1, 2, 2, 2],
                      'hit_id': [1, 10, 3, 9, 4, 8, 5, 7, 6]}).set_index('hit_id')


def test_gen_seg_track_sequential():
    assert_array_equal(list(gen_seg_track_sequential(_hits)), [(1, 7), (3, 8), (8, 6), (9, 5)])


def test_gen_seg_track_layered():
    assert_array_equal(gen_seg_track_layered(_hits), [(3, 8), (8, 6), (9, 5)])
