import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from datasets.simple import SimpleEventGenerator


def test_gen_directions_in_cone():
    v = SimpleEventGenerator(seed=1).gen_directions_in_cone(13)
    assert v.shape == (13, 3)
    assert_array_equal(SimpleEventGenerator(0).gen_directions_in_cone(100),
                       [[0, 0, 1]] * 100)
    assert_allclose(SimpleEventGenerator(180, 1).gen_directions_in_cone(100_000).mean(axis=0),
                    [0, 0, 0], atol=0.01)


def test_run_straight_track():
    g = SimpleEventGenerator(field_strength=0.)
    assert_array_equal(g.run_straight_track(np.array([0, 0, 1])),
                       [[0, 0, 0.5], [0, 0, 1], [0, 0, 1.5], [0, 0, 2],
                        [0, 0, 2.5], [0, 0, 3], [0, 0, 3.5], [0, 0, 4]])
    assert_array_equal(g.run_straight_track(np.array([1, -1, 1])),
                       [[0.5, -0.5, 0.5], [1, -1, 1], [1.5, -1.5, 1.5], [2, -2, 2],
                        [2.5, -2.5, 2.5], [3, -3, 3], [3.5, -3.5, 3.5], [4, -4, 4]])


def test_run_curved_track():
    g = SimpleEventGenerator(field_strength=1.)
    assert_array_equal(g.run_curved_track(np.array([0, 0, 1]), 123.),
                       [[0, 0, 0.5], [0, 0, 1], [0, 0, 1.5], [0, 0, 2],
                        [0, 0, 2.5], [0, 0, 3], [0, 0, 3.5], [0, 0, 4]])
    r2 = np.sqrt(2) / 2
    assert_allclose(g.run_curved_track(np.array([0, -1, 1 / np.pi]), 1.0),
                    [[1 - r2, -r2, 0.5], [1, -1, 1], [1 + r2, -r2, 1.5], [2, 0, 2],
                     [1 + r2, r2, 2.5], [1, 1, 3], [1 - r2, r2, 3.5], [0, 0, 4]], atol=1e-14)
    assert_allclose(g.run_curved_track(np.array([0, -1, 1 / np.pi]), -1.0),
                    [[- 1 + r2, -r2, 0.5], [-1, -1, 1], [-1 - r2, -r2, 1.5], [-2, 0, 2],
                     [-1 - r2, r2, 2.5], [-1, 1, 3], [-1 + r2, r2, 3.5], [0, 0, 4]], atol=1e-14)


def test_gen_event_nonmagnetic():
    a = np.array([[0, 0, 1], [1, -1, 1]])
    hits, seg = SimpleEventGenerator(field_strength=0., noisiness=0, seed=1).gen_event(a, np.ones(2))
    assert_array_equal(hits.index, range(16))
    assert list(hits.columns) == ['x', 'y', 'z', 'layer', 'track', 'charge']
    assert_allclose(hits[['x', 'y']].values[:8], [[0, 0]] * 8, atol=0.03)
    assert_allclose(hits[['x', 'y']].values[8:16],
                    [[0.5, -0.5], [1, -1], [1.5, -1.5], [2, -2],
                     [2.5, -2.5], [3, -3], [3.5, -3.5], [4, -4]], atol=0.03)
    assert_array_equal(hits[['z', 'layer', 'track', 'charge']].values,
                       [[0.5, 0, 0, 1], [1., 1, 0, 1], [1.5, 2, 0, 1], [2., 3, 0, 1],
                        [2.5, 4, 0, 1], [3., 5, 0, 1], [3.5, 6, 0, 1], [4., 7, 0, 1],
                        [0.5, 0, 1, 1], [1., 1, 1, 1], [1.5, 2, 1, 1], [2., 3, 1, 1],
                        [2.5, 4, 1, 1], [3., 5, 1, 1], [3.5, 6, 1, 1], [4., 7, 1, 1]])
    assert_array_equal(seg, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
                             [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15]])

    hits, seg = SimpleEventGenerator(field_strength=11., noisiness=0, seed=1).gen_event(a, np.zeros(2))
    assert_array_equal(hits.index, range(16))
    assert list(hits.columns) == ['x', 'y', 'z', 'layer', 'track', 'charge']
    assert_allclose(hits[['x', 'y']].values[:8], [[0, 0]] * 8, atol=0.03)
    assert_allclose(hits[['x', 'y']].values[8:16],
                    [[0.5, -0.5], [1, -1], [1.5, -1.5], [2, -2],
                     [2.5, -2.5], [3, -3], [3.5, -3.5], [4, -4]], atol=0.03)
    assert_array_equal(hits[['z', 'layer', 'track', 'charge']].values,
                       [[0.5, 0, 0, 0], [1., 1, 0, 0], [1.5, 2, 0, 0], [2., 3, 0, 0],
                        [2.5, 4, 0, 0], [3., 5, 0, 0], [3.5, 6, 0, 0], [4., 7, 0, 0],
                        [0.5, 0, 1, 0], [1., 1, 1, 0], [1.5, 2, 1, 0], [2., 3, 1, 0],
                        [2.5, 4, 1, 0], [3., 5, 1, 0], [3.5, 6, 1, 0], [4., 7, 1, 0]])
    assert_array_equal(seg, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
                             [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15]])


def test_gen_event_magnetic():
    momenta = np.array([[0, 0, 1], [0, -1, 1 / np.pi]])
    hits, seg = SimpleEventGenerator(seed=1).gen_event(momenta, np.ones(2))
    assert_array_equal(hits.index, range(19))
    assert list(hits.columns) == ['x', 'y', 'z', 'layer', 'track', 'charge']
    r2 = np.sqrt(2) / 2
    side = 4. * np.sin(15 / 180 * np.pi)
    assert_allclose(hits[['x', 'y']].values[:8], [[0, 0]] * 8, atol=0.03)
    assert_allclose(hits[['x', 'y']].values[8:16], [[1 - r2, -r2], [1, -1], [1 + r2, -r2], [2, 0],
                                                    [1 + r2, r2], [1, 1], [1 - r2, r2], [0, 0]], atol=0.03)
    assert (hits[['x', 'y']].values[16:] <= side).all()
    assert (hits[['x', 'y']].values[16:] >= -side).all()

    assert_array_equal(hits[['z', 'layer', 'track', 'charge']].values[:16],
                       [[0.5, 0, 0, 1], [1., 1, 0, 1], [1.5, 2, 0, 1], [2., 3, 0, 1],
                        [2.5, 4, 0, 1], [3., 5, 0, 1], [3.5, 6, 0, 1], [4., 7, 0, 1],
                        [0.5, 0, 1, 1], [1., 1, 1, 1], [1.5, 2, 1, 1], [2., 3, 1, 1],
                        [2.5, 4, 1, 1], [3., 5, 1, 1], [3.5, 6, 1, 1], [4., 7, 1, 1]])
    assert np.isin(hits['layer'].values[16:], np.arange(8)).all()
    assert_array_equal(0.5 + hits.layer * 0.5, hits.z)
    assert_array_equal(hits.iloc[16:].track, - np.ones(3))
    assert hits.iloc[16:].charge.isna().all()

    assert_array_equal(seg, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
                             [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15]])

    hits, seg = SimpleEventGenerator(field_strength=-1, noisiness=0, seed=1).gen_event(momenta,
                                                                  -np.ones(2))  # negative field and negative charge!
    assert_array_equal(hits.index, range(16))
    assert list(hits.columns) == ['x', 'y', 'z', 'layer', 'track', 'charge']
    assert_allclose(hits[['x', 'y']].values, [[0, 0]] * 8 +
                    [[1 - r2, -r2], [1, -1], [1 + r2, -r2], [2, 0],
                     [1 + r2, r2], [1, 1], [1 - r2, r2], [0, 0]], atol=0.03)
    assert_array_equal(hits[['z', 'layer', 'track', 'charge']].values,
                       [[0.5, 0, 0, -1], [1., 1, 0, -1], [1.5, 2, 0, -1], [2., 3, 0, -1],
                        [2.5, 4, 0, -1], [3., 5, 0, -1], [3.5, 6, 0, -1], [4., 7, 0, -1],
                        [0.5, 0, 1, -1], [1., 1, 1, -1], [1.5, 2, 1, -1], [2., 3, 1, -1],
                        [2.5, 4, 1, -1], [3., 5, 1, -1], [3.5, 6, 1, -1], [4., 7, 1, -1]])
    assert_array_equal(seg, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
                             [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15]])


def test_gen_many_events():
    data = list(SimpleEventGenerator(noisiness=0).gen_many_events(5, 7))
    assert len(data) == 5
    for hits, seg in data:
        assert len(hits.index) == 8 * 7
        assert len(hits.columns) == 6
        assert seg.shape == (7 * 7, 2)
