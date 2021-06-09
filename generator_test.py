import numpy as np
from numpy.ma.testutils import assert_array_approx_equal
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_allclose

from generator import SimpleEventGenerator


def test_gen_one_track():
    v = SimpleEventGenerator().gen_one_track()
    assert len(v) == 3


def test_gen_event_tracks():
    a = SimpleEventGenerator().gen_event_tracks(2)
    assert a.shape == (2, 3)
    assert a.dtype == float


def test_gen_event_hits():
    a = np.array([[1, 0, 0], [1, 1, -1]])
    hits = SimpleEventGenerator().gen_event_hits(a)
    assert_array_equal(hits.index, range(16))
    assert list(hits.columns) == ['x', 'y', 'z', 'layer', 'track']
    assert_allclose(hits[['y', 'z']].values,
                              [[0, 0], [0.5, -0.5], [0, 0], [1, -1],
                               [0, 0], [1.5, -1.5], [0, 0], [2, -2],
                               [0, 0], [2.5, -2.5], [0, 0], [3, -3],
                               [0, 0], [3.5, -3.5], [0, 0], [4, -4]], atol=0.03)
    assert_array_equal(hits[['x', 'layer', 'track']].values,
                       [[0.5, 0, 0], [0.5, 0, 1], [1, 1, 0], [1, 1, 1],
                        [1.5, 2, 0], [1.5, 2, 1], [2, 3, 0], [2, 3, 1],
                        [2.5, 4, 0], [2.5, 4, 1], [3, 5, 0], [3, 5, 1],
                        [3.5, 6, 0], [3.5, 6, 1], [4, 7, 0], [4, 7, 1]])


def test_gen_many_events():
    data = list(SimpleEventGenerator().gen_many_events(5, 7))
    assert len(data) == 5
    for event in data:
        assert len(event.index) == 8 * 7
        assert len(event.columns) == 5
