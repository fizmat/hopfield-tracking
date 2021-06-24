import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from generator import SimpleEventGenerator


def test_gen_directions_in_cone():
    v = SimpleEventGenerator(seed=1).gen_directions_in_cone(13)
    assert v.shape == (13, 3)
    assert_array_equal(SimpleEventGenerator(0).gen_directions_in_cone(100),
                       [[1, 0, 0]] * 100)
    assert_allclose(SimpleEventGenerator(180, 1).gen_directions_in_cone(100_000).mean(axis=0),
                    [0, 0, 0], atol=0.01)


def test_run_straight_track():
    g = SimpleEventGenerator(field_strength=0.)
    assert_array_equal(g.run_straight_track(np.array([1, 0, 0])),
                       [[0.5, 0, 0], [1, 0, 0], [1.5, 0, 0], [2, 0, 0],
                        [2.5, 0, 0], [3, 0, 0], [3.5, 0, 0], [4, 0, 0]])
    assert_array_equal(g.run_straight_track(np.array([1, 1, -1])),
                       [[0.5, 0.5, -0.5], [1, 1, -1], [1.5, 1.5, -1.5], [2, 2, -2],
                        [2.5, 2.5, -2.5], [3, 3, -3], [3.5, 3.5, -3.5], [4, 4, -4]])


def test_run_curved_track():
    g = SimpleEventGenerator(field_strength=1.)
    assert_array_equal(g.run_curved_track(np.array([1, 0, 0])),
                       [[0.5, 0, 0], [1, 0, 0], [1.5, 0, 0], [2, 0, 0],
                        [2.5, 0, 0], [3, 0, 0], [3.5, 0, 0], [4, 0, 0]])
    r2 = np.sqrt(2) / 2
    assert_allclose(g.run_curved_track(np.array([1 / np.pi, 0, -1])),
                    [[0.5, 1 - r2, -r2], [1, 1, -1], [1.5, 1 + r2, -r2], [2, 2, 0],
                     [2.5, 1 + r2, r2], [3, 1, 1], [3.5, 1 - r2, r2], [4, 0, 0]], atol=1e-14)


def test_gen_event_nonmagnetic():
    a = np.array([[1, 0, 0], [1, 1, -1]])
    hits, seg = SimpleEventGenerator(field_strength=0.).gen_event(a)
    assert_array_equal(hits.index, range(16))
    assert list(hits.columns) == ['x', 'y', 'z', 'layer', 'track']
    assert_allclose(hits[['y', 'z']].values, [[0, 0]] * 8 +
                    [[0.5, -0.5], [1, -1], [1.5, -1.5], [2, -2],
                     [2.5, -2.5], [3, -3], [3.5, -3.5], [4, -4]], atol=0.03)
    assert_array_equal(hits[['x', 'layer', 'track']].values,
                       [[0.5, 0, 0], [1., 1, 0], [1.5, 2, 0], [2., 3, 0],
                        [2.5, 4, 0], [3., 5, 0], [3.5, 6, 0], [4., 7, 0],
                        [0.5, 0, 1], [1., 1, 1], [1.5, 2, 1], [2., 3, 1],
                        [2.5, 4, 1], [3., 5, 1], [3.5, 6, 1], [4., 7, 1]])
    assert_array_equal(seg, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
                             [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15]])


def test_gen_event_magnetic():
    momenta = np.array([[1, 0, 0], [1 / np.pi, 0, -1]])
    hits, seg = SimpleEventGenerator().gen_event(momenta)
    assert_array_equal(hits.index, range(16))
    assert list(hits.columns) == ['x', 'y', 'z', 'layer', 'track']
    r2 = np.sqrt(2) / 2
    assert_allclose(hits[['y', 'z']].values, [[0, 0]] * 8 +
                    [[1 - r2, -r2], [1, -1], [1 + r2, -r2], [2, 0],
                     [1 + r2, r2], [1, 1], [1 - r2, r2], [0, 0]], atol=0.03)
    assert_array_equal(hits[['x', 'layer', 'track']].values,
                       [[0.5, 0, 0], [1., 1, 0], [1.5, 2, 0], [2., 3, 0],
                        [2.5, 4, 0], [3., 5, 0], [3.5, 6, 0], [4., 7, 0],
                        [0.5, 0, 1], [1., 1, 1], [1.5, 2, 1], [2., 3, 1],
                        [2.5, 4, 1], [3., 5, 1], [3.5, 6, 1], [4., 7, 1]])
    assert_array_equal(seg, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
                             [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15]])


def test_gen_many_events():
    data = list(SimpleEventGenerator().gen_many_events(5, 7))
    assert len(data) == 5
    for hits, seg in data:
        assert len(hits.index) == 8 * 7
        assert len(hits.columns) == 5
        assert seg.shape == (7 * 7, 2)
