import numpy as np
from numpy.testing import assert_allclose

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
    assert len(hits) == 8
    assert_allclose(hits[0], [[0.5, 0, 0], [0.5, 0.5, -0.5]], atol=0.02)
    assert_allclose(hits[1], [[1, 0, 0], [1, 1, -1]], atol=0.02)
    assert_allclose(hits[2], [[1.5, 0, 0], [1.5, 1.5, -1.5]], atol=0.02)
    assert_allclose(hits[3], [[2, 0, 0], [2, 2, -2]], atol=0.02)
    assert_allclose(hits[4], [[2.5, 0, 0], [2.5, 2.5, -2.5]], atol=0.02)
    assert_allclose(hits[5], [[3, 0, 0], [3, 3, -3]], atol=0.02)
    assert_allclose(hits[6], [[3.5, 0, 0], [3.5, 3.5, -3.5]], atol=0.02)
    assert_allclose(hits[7], [[4, 0, 0], [4, 4, -4]], atol=0.02)


def test_gen_many_events():
    data = list(SimpleEventGenerator().gen_many_events(5, 7))
    assert len(data) == 5
    for event in data:
        assert len(event) == 8
        for layer in event:
            assert layer.shape == (7, 3)
