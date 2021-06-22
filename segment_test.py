import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal, assert_array_equal
from pytest import approx

from segment import energy, energy_gradients, \
    gen_segments_layer, gen_segments_all


def test_gen_segments_layer():
    a = np.arange(2)
    b = np.arange(4)
    assert_array_equal(gen_segments_layer(a, b), [[0, 0], [0, 1], [0, 2], [0, 3],
                                                  [1, 0], [1, 1], [1, 2], [1, 3]])


def test_gen_segment_all():
    df = pd.DataFrame({'x': 0, 'y': 0, 'z': 0, 'layer': [0, 0, 1, 1, 1, 2], 'track': 0})
    v1, v2 = gen_segments_all(df)
    assert_array_equal(v1, [[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4]])
    assert_array_equal(v2, [[2, 5], [3, 5], [4, 5]])


def test_energy_empty():
    assert energy(np.empty(0), [])([]) == 0


def test_energy_gradients_empty():
    assert energy_gradients(np.empty(0), [])([]) == ([], [], [])


def test_energy_two_hits():
    # vertex count energy only
    pos = [np.array([[0., 0]]),
           np.array([[1, 0]])]
    seg = [np.array([[0, 0]])]
    v = np.array([0.])
    assert energy(pos, seg, drop_gradients_on_self=False)([v]) == 2
    v = np.array([1.])
    assert energy(pos, seg, drop_gradients_on_self=False)([v]) == 0.5


def test_energy_gradients_two_hits():
    # vertex count energy only
    pos = np.array([[0., 0], [1, 0]])
    seg = [np.array([[0, 1]])]
    v = np.array([0.])
    eg = energy_gradients(pos, seg, drop_gradients_on_self=False)([v])
    assert [len(e) for e in eg] == [1, 1, 1]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], [0])
    assert_array_almost_equal(eng[0], [-2])
    assert_array_almost_equal(efg[0], [0])
    v = np.array([1.])
    eg = energy_gradients(pos, seg, drop_gradients_on_self=False)([v])
    assert [len(e) for e in eg] == [1, 1, 1]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], [0])
    assert_array_almost_equal(eng[0], [-1])
    assert_array_almost_equal(efg[0], [0])

    # eg = energy_gradients(pos, seg, drop_gradients_on_self=True)(v)
    # assert [len(e) for e in eg] == [1, 1, 1]
    # ecg, eng, efg = eg
    # assert_array_almost_equal(ecg[0], [0])
    # assert_array_almost_equal(eng[0], [-2])
    # assert_array_almost_equal(efg[0], [0])
    # v = [np.array([1.])]
    # eg = energy_gradients(pos, seg, drop_gradients_on_self=True)(v)
    # assert [len(e) for e in eg] == [1, 1, 1]
    # ecg, eng, efg = eg
    # assert_array_almost_equal(ecg[0], [0])
    # assert_array_almost_equal(eng[0], [-2])
    # assert_array_almost_equal(efg[0], [0])


def test_energy_four_hits():
    pos = np.array([[0., 0], [0, 1], [1, 0], [1, 1]])
    seg = [np.array([[0, 2], [0, 3], [1, 2], [1, 3]])]
    e = energy(pos, seg, drop_gradients_on_self=False)
    v = np.zeros(4)
    assert e(v) == 8  # bad vertex count
    v = np.ones(4)
    assert e(v) == 4  # many forks


def test_energy_gradients_four_hits():
    # vertex count energy only
    pos = np.array([[0., 0], [0, 1], [1, 0], [1, 1]])
    seg = [np.array([[0, 2], [0, 3], [1, 2], [1, 3]])]
    v = np.zeros(4)
    eg = energy_gradients(pos, seg, drop_gradients_on_self=False)([v])
    assert [len(e) for e in eg] == [1, 1, 1]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], np.zeros(4))
    assert_array_almost_equal(eng[0], np.full(4, -4))
    assert_array_almost_equal(efg[0], np.zeros(4))
    v = np.ones(4)
    eg = energy_gradients(pos, seg, drop_gradients_on_self=False)([v])
    assert [len(e) for e in eg] == [1, 1, 1]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], np.zeros(4))  # short tracks, no cosines yet
    assert_array_almost_equal(eng[0], np.zeros(4))  # exactly 4 active for 4 vertices
    assert_array_almost_equal(efg[0], np.full(4, 2))  # forks and joins

    v = np.zeros(4)
    eg = energy_gradients(pos, seg, drop_gradients_on_self=True)([v])
    assert [len(e) for e in eg] == [1, 1, 1]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], np.zeros(4))
    assert_array_almost_equal(eng[0], np.full(4, -4))
    assert_array_almost_equal(efg[0], np.zeros(4))
    v = np.ones(4)
    eg = energy_gradients(pos, seg, drop_gradients_on_self=True)([v])
    assert [len(e) for e in eg] == [1, 1, 1]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], np.zeros(4))  # short tracks, no cosines yet
    assert_array_almost_equal(eng[0], np.full(4, -1))  # 4 active for 4 vertices
    assert_array_almost_equal(efg[0], np.full(4, 2))  # forks and joins


def test_energy_one_track():
    v = np.array([1., 1.])
    pos = np.array([[0., 0], [1., 0], [2., 0]])
    seg = [np.array([[0, 1]]), np.array([[1, 2]])]
    e = energy(pos, seg, drop_gradients_on_self=False)
    assert e(v) == 0.5 + 0 - 0.5

    v = np.array([1., 1.])
    pos = np.array([[0., 0], [1., 0], [2., 1]])
    e = energy(pos, seg, drop_gradients_on_self=False)
    assert e(v).item() == approx(0.5 + 0 - 1. / 8)


def test_energy_gradients_one_track():
    v = [np.array([1.]), np.array([1.])]
    # straight track
    pos = np.array([[0., 0], [1., 0], [2., 0]])
    seg = [np.array([[0, 1]]), np.array([[1, 2]])]
    eg = energy_gradients(pos, seg, drop_gradients_on_self=False)(v)
    assert [len(e) for e in eg] == [2, 2, 2]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], [-0.5])  # reward for straight track
    assert_array_almost_equal(eng[0], [-1])  # 2 active for 3 hits, wants more
    assert_array_almost_equal(efg[0], [0])  # no forks or joins possible
    assert_array_almost_equal(ecg[1], [-0.5])
    assert_array_almost_equal(eng[1], [-1])
    assert_array_almost_equal(efg[1], [0])

    # curved track
    pos = np.array([[0., 0], [1., 0], [2., 1]])
    eg = energy_gradients(pos, seg, drop_gradients_on_self=False)(v)
    assert [len(e) for e in eg] == [2, 2, 2]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], [-1. / 8])
    assert_array_almost_equal(eng[0], [-1.])
    assert_array_almost_equal(efg[0], [0])
    assert_array_almost_equal(ecg[1], [-1. / 8])
    assert_array_almost_equal(eng[1], [-1.])
    assert_array_almost_equal(efg[1], [0])

    # straight track
    pos = np.array([[0., 0], [1., 0], [2., 0]])
    eg = energy_gradients(pos, seg, drop_gradients_on_self=True)(v)
    assert [len(e) for e in eg] == [2, 2, 2]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], [-0.5])  # reward for straight track
    assert_array_almost_equal(eng[0], [-2])  # 2 - itself = 1 active for 3 hits, wants more
    assert_array_almost_equal(efg[0], [0])  # no forks or joins possible
    assert_array_almost_equal(ecg[1], [-0.5])
    assert_array_almost_equal(eng[1], [-2])
    assert_array_almost_equal(efg[1], [0])

    # curved track
    pos = np.array([[0., 0], [1., 0], [2., 1]])
    eg = energy_gradients(pos, seg, drop_gradients_on_self=True)(v)
    assert [len(e) for e in eg] == [2, 2, 2]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], [-1. / 8])
    assert_array_almost_equal(eng[0], [-2])
    assert_array_almost_equal(efg[0], [0])
    assert_array_almost_equal(ecg[1], [-1. / 8])
    assert_array_almost_equal(eng[1], [-2])
    assert_array_almost_equal(efg[1], [0])
