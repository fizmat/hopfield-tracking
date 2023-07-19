import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from metrics.tracks import build_segmented_tracks, enumerate_segmented_track, found_tracks, found_crosses, \
    track_metrics

_hits = pd.DataFrame({'track': [1, -1, 3, 5, -1, 3, 5, 1, 3],
                      'layer': [0, 0, 0, 1, 1, 1, 2, 2, 2]})

_seg = np.array([(0, 3), (0, 4), (0, 5),
                 (1, 3), (1, 4), (1, 5),
                 (2, 3), (2, 4), (2, 5),
                 (3, 6), (3, 7), (3, 8),
                 (4, 6), (4, 7), (4, 8),
                 (5, 6), (5, 7), (5, 8)])

_tracks = {1: [], 3: [(2, 5), (5, 8)], 5: [(3, 6)]}
_tseg = np.array([(2, 5), (5, 8), (3, 6)])


def test_build_segmented_tracks():
    assert build_segmented_tracks(_hits) == _tracks


def test_enumerate_segmented_track():
    assert enumerate_segmented_track([(2, 5), (5, 8)], _seg) == [8, 17]
    assert enumerate_segmented_track(
        [(21, 44), (44, 79), (79, 123), (123, 164), (164, 190), (190, 209), (209, 228), (228, 242)],
        np.array([(21, 44), (44, 79), (79, 123), (123, 164), (164, 190), (190, 209), (209, 228), (228, 242)])
    ) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert enumerate_segmented_track([(0, 1), (1, 3)], np.array([(0, 1), (0, 2), (1, 3)])) == [0, 2]


def test_found_tracks():
    all_tracks = list(_tracks.values())
    assert found_tracks(_seg, np.zeros(len(_seg)), all_tracks) == 1
    assert found_tracks(_seg, np.ones(len(_seg)), all_tracks) == 3
    assert found_tracks(_seg, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 0]), all_tracks) == 2
    assert found_tracks(_seg, np.array([0, 0, 0, 0, 0, 0, 0, 0, 1,
                                        0, 0, 0, 0, 0, 0, 0, 0, 1]), all_tracks) == 2
    assert found_tracks(_seg, np.array([0, 0, 0, 0, 0, 0, 0, 0, 1,
                                        1, 0, 0, 0, 0, 0, 0, 0, 1]), all_tracks) == 3


def test_found_crosses():
    assert found_crosses(_seg, np.zeros(len(_seg))) == 0
    assert found_crosses(_seg, np.ones(len(_seg))) == ((3 * 3) + (3 * 3)) * 2
    assert found_crosses(_seg, np.array([1, 0, 0, 1, 0, 0, 1, 0, 0,
                                         1, 0, 0, 1, 0, 0, 1, 0, 0])) == 3 + 3
    assert found_crosses(_seg, np.array([1, 0, 0, 0, 1, 0, 0, 0, 1,
                                         1, 0, 0, 0, 1, 0, 0, 0, 1])) == 0
    assert found_crosses(_seg, np.array([1, 0, 0, 0, 1, 0, 0, 0, 1,
                                         1, 0, 0, 0, 1, 0, 0, 1, 1])) == 2
    assert found_crosses(_seg, np.array([1, 0, 0, 0, 1, 0, 0, 0, 1,
                                         1, 0, 0, 0, 1, 0, 0, 1, 0])) == 1
    _seg2 = np.array([(0, 2), (0, 3), (1, 2), (1, 3)])
    assert found_crosses(_seg2, np.array((0, 0, 0, 0))) == 0
    assert found_crosses(_seg2, np.array((1, 0, 0, 1))) == 0
    assert found_crosses(_seg2, np.array((0, 1, 1, 0))) == 0
    assert found_crosses(_seg2, np.array((1, 0, 1, 0))) == 1
    assert found_crosses(_seg2, np.array((1, 1, 0, 0))) == 1
    assert found_crosses(_seg2, np.array((1, 1, 1, 1))) == 4


def test_track_metrics():
    assert track_metrics(_hits, _seg, _tseg, np.zeros(len(_seg)), np.full(len(_seg), False)) == {'trackml': 0.0}
    #{'reds': 0, 'tracks': 1, 'crosses': 0}
    assert track_metrics(_hits, _seg, _tseg, np.ones(len(_seg)), np.full(len(_seg), True)) == {'trackml': 0.0}
    #{'reds': 15, 'tracks': 3, 'crosses': 36}
    act = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1,
                    1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(bool)
    assert track_metrics(_hits, _seg, _tseg, act, act.astype(bool)) == {'trackml': 0.7142857142857142}
                                                                                            # {'reds': 3,
                                                                                            #  'tracks': 3,
                                                                                            #  'crosses': 0}
    act = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1,
                    1, 0, 0, 0, 0, 0, 0, 0, 1])
    assert track_metrics(_hits, _seg, _tseg, act, act.astype(bool)) == {'trackml': 0.7142857142857142}
                                                                                            # {'reds': 0,
                                                                                            #  'tracks': 3,
                                                                                            #  'crosses': 0}
