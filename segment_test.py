import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal, assert_array_equal
from pytest import approx
from scipy.sparse import csr_matrix

from segment import number_of_used_vertices_energy, curvature_energy, \
    count_segments, energy, curvature_energy_matrix, curvature_energy_gradient, energy_gradients, \
    number_of_used_vertices_energy_gradient, \
    curvature_energy_pairwise, gen_segments_layer, gen_segments_all, fork_energy_matrix, layer_energy, \
    layer_energy_gradient, join_energy_matrix


def test_gen_segments_layer():
    a = np.arange(2)
    b = np.arange(4)
    assert_array_equal(gen_segments_layer(a, b),
                       [[0, 0, 0, 0, 1, 1, 1, 1],
                        [0, 1, 2, 3, 0, 1, 2, 3]])


def test_gen_segment_all():
    df = pd.DataFrame({'x': 0, 'y': 0, 'z': 0, 'layer': [0, 0, 1, 1, 1, 2], 'track': 0})
    v1, v2 = gen_segments_all(df)
    assert_array_equal(v1, [[0, 0, 0, 1, 1, 1], [2, 3, 4, 2, 3, 4]])
    assert_array_equal(v2, [[2, 3, 4], [5, 5, 5]])


def test_fork_matrix():
    segments = np.array([[0, 0, 1, 1], [2, 3, 2, 3]])
    m = fork_energy_matrix(segments)
    assert_array_equal(m.todense(), [[0, 1, 0, 0],
                                     [1, 0, 0, 0],
                                     [0, 0, 0, 1],
                                     [0, 0, 1, 0]])


def test_join_matrix():
    segments = np.array([[0, 0, 1, 1], [2, 3, 2, 3]])
    m = join_energy_matrix(segments)
    assert_array_equal(m.todense(), [[0, 0, 1, 0],
                                     [0, 0, 0, 1],
                                     [1, 0, 0, 0],
                                     [0, 1, 0, 0]])


def test_layer_energy():
    m = np.array([[0, 1], [1, 0]])
    assert layer_energy(m, np.array([0, 0])) == 0
    assert layer_energy(m, np.array([1, 0])) == 0
    assert layer_energy(csr_matrix(m), np.array([0, 1])) == 0
    assert layer_energy(m, np.array([1, 1])) == 1
    assert layer_energy(csr_matrix(m), np.array([0.5, 1])) == 0.5


def test_layer_energy_gradient():
    m = np.array([[0, 1], [1, 0]])
    assert_array_almost_equal(layer_energy_gradient(m, np.array([0, 0])),
                              np.array([0, 0]))
    assert_array_almost_equal(layer_energy_gradient(m, np.array([1, 0])),
                              np.array([0, 1]))
    assert_array_almost_equal(layer_energy_gradient(csr_matrix(m), np.array([0, 1])),
                              np.array([1, 0]))
    assert_array_almost_equal(layer_energy_gradient(m, np.array([1, 1])),
                              np.array([1, 1]))
    assert_array_almost_equal(layer_energy_gradient(csr_matrix(m), np.array([0.5, 1])),
                              np.array([1, 0.5]))


def test_count_segments():
    assert count_segments([]) == 0
    assert count_segments([np.zeros((2, 4))]) == 0
    assert count_segments([np.ones((3, 2)), np.ones((2, 4))]) == 14
    assert count_segments((
        np.array([[0.5, 0.2], [0, 0.1]]),
        np.array([[0.1], [0.2]])
    )) == approx(1.1)


def test_number_of_used_vertices_energy():
    assert number_of_used_vertices_energy(0, np.array([0])) == 0
    assert number_of_used_vertices_energy(6, np.array([0])) == 18
    assert number_of_used_vertices_energy(6, np.array([1])) == 12.5
    assert number_of_used_vertices_energy(6, np.array([2])) == 8
    assert number_of_used_vertices_energy(6, np.array([3])) == 4.5
    assert number_of_used_vertices_energy(6, np.array([4])) == 2
    assert number_of_used_vertices_energy(6, np.array([5])) == 0.5
    assert number_of_used_vertices_energy(6, np.array([6])) == 0
    assert number_of_used_vertices_energy(6, np.array([7])) == 0.5
    assert number_of_used_vertices_energy(6, np.array([9])) == 4.5


def test_count_vertices_gradient():
    assert number_of_used_vertices_energy_gradient(0, 0) == 0
    assert number_of_used_vertices_energy_gradient(6, 0) == -6
    assert number_of_used_vertices_energy_gradient(6, 2.) == -4
    assert number_of_used_vertices_energy_gradient(6, 6.) == 0
    assert number_of_used_vertices_energy_gradient(6, 8.) == 2


def test_curvature_energy_pairwise():
    a = np.array([[0., 0], [0., 1], [0., 2]])
    b = np.array([[1., 0], [1., 1], [1., 2]])
    c = np.array([[2., 0], [2, 0], [3, 4]])
    w = curvature_energy_pairwise(a, b, c)
    assert_array_almost_equal(w, [-1. / 2, -1. / 8, -1. / 16])


def test_curvature_energy_matrix():
    pos = np.array([[0., 0], [1., 0], [2., 0], [2, 1], [2, -1], [3, 0], [3, 2]])
    s1 = np.array([[0], [1]])
    s2 = np.array([[1, 1, 1, 1, 1], [2, 3, 4, 5, 6]])
    w = curvature_energy_matrix(pos, s1, s2)
    assert_array_equal(w.row, [0, 0, 0, 0, 0])
    assert_array_equal(w.col, [0, 1, 2, 3, 4])
    assert_array_almost_equal(w.data, [-1. / 2, -1. / 8, -1. / 8, -1. / 4, -1. / 16])


def test_curvature_energy():
    pos = np.array([[0., 0], [1., 0], [2., 0], [2, 1], [2, -1], [3, 0], [3, 2]])
    s1 = np.array([[0], [1]])
    s2 = np.array([[1, 1, 1, 1, 1], [2, 3, 4, 5, 6]])
    w = curvature_energy_matrix(pos, s1, s2)
    first = np.array([1])
    assert curvature_energy(w, first, np.array([1, 0, 0, 0, 0])) == - 0.5
    assert curvature_energy(w, first, np.array([0, 1, 0, 0, 0])) == approx(- 1. / 8)
    assert curvature_energy(w, first, np.array([0, 0, 1, 0, 0])) == approx(- 1. / 8)
    assert curvature_energy(w, first, np.array([0, 0, 0, 1, 0])) == approx(- 1. / 4)
    assert curvature_energy(w, first, np.array([0, 0, 0, 0, 1])) == approx(- 1. / 16)
    assert curvature_energy(w, first, np.array([1, 1, 1, 1, 1])) == approx(- 17. / 16)
    assert curvature_energy(w, first, np.array([.1, .1, .1, .1, .1])) == approx(- 1.7 / 16)


def test_curvature_energy_gradient():
    pos = np.array([[0., 0], [1., 0], [2., 0], [2, 1], [2, -1], [3, 0], [3, 2]])
    s1 = np.array([[0], [1]])
    s2 = np.array([[1, 1, 1, 1, 1], [2, 3, 4, 5, 6]])
    w = curvature_energy_matrix(pos, s1, s2)
    first = np.array([1.])
    second = np.array([1., 0, 0, 0, 0])
    g1, g2 = curvature_energy_gradient(w, first, second)
    assert_array_almost_equal(g1, np.array([-0.5]))
    assert_array_almost_equal(g2, np.array([- 0.5, -1. / 8, -1. / 8, -1. / 4, -1. / 16]))


def test_energy_empty():
    assert energy(np.empty(0), [])([]) == 0


def test_energy_gradients_empty():
    assert energy_gradients(np.empty(0), [])([]) == ([], [], [])


def test_energy_two_hits():
    # vertex count energy only
    pos = [np.array([[0., 0]]),
           np.array([[1, 0]])]
    seg = [np.array([[0], [0]])]
    v = np.array([[0.]])
    assert energy(pos, seg)([v]) == 2
    v = np.array([[1.]])
    assert energy(pos, seg)([v]) == 0.5


def test_energy_gradients_two_hits():
    # vertex count energy only
    pos = np.array([[0., 0], [1, 0]])
    seg = [np.array([[0], [1]])]
    v = np.array([[0.]])
    eg = energy_gradients(pos, seg, drop_gradients_on_self=False)([v])
    assert [len(e) for e in eg] == [1, 1, 1]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], [[0]])
    assert_array_almost_equal(eng[0], [[-2]])
    assert_array_almost_equal(efg[0], [[0]])
    v = np.array([[1.]])
    eg = energy_gradients(pos, seg, drop_gradients_on_self=False)([v])
    assert [len(e) for e in eg] == [1, 1, 1]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], [[0]])
    assert_array_almost_equal(eng[0], [[-1]])
    assert_array_almost_equal(efg[0], [[0]])

    eg = energy_gradients(pos, seg, drop_gradients_on_self=True)([v])
    assert [len(e) for e in eg] == [1, 1, 1]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], [[0]])
    assert_array_almost_equal(eng[0], [[-2]])
    assert_array_almost_equal(efg[0], [[0]])
    v = np.array([[1.]])
    eg = energy_gradients(pos, seg, drop_gradients_on_self=True)([v])
    assert [len(e) for e in eg] == [1, 1, 1]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], [[0]])
    assert_array_almost_equal(eng[0], [[-2]])
    assert_array_almost_equal(efg[0], [[0]])


def test_energy_four_hits():
    pos = np.array([[0., 0], [0, 1], [1, 0], [1, 1]])
    seg = [np.array([[0, 0, 1, 1], [2, 3, 2, 3]])]
    e = energy(pos, seg)
    v = np.zeros(4)
    assert e([v]) == 8  # bad vertex count
    v = np.ones(4)
    assert e([v]) == 4  # many forks


def test_energy_gradients_four_hits():
    # vertex count energy only
    pos = np.array([[0., 0], [0, 1], [1, 0], [1, 1]])
    seg = [np.array([[0, 0, 1, 1], [2, 3, 2, 3]])]
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
    v = [np.array([[1.]]), np.array([[1.]])]
    pos = np.array([[0., 0], [1., 0], [2., 0]])
    seg = [np.array([[0], [1]]), np.array([[1], [2]])]
    e = energy(pos, seg)
    assert e(v) == 0.5 + 0 - 0.5

    v = [np.array([[1.]]), np.array([[1.]])]
    pos = np.array([[0., 0], [1., 0], [2., 1]])
    e = energy(pos, seg)
    assert e(v).item() == approx(0.5 + 0 - 1. / 8)


def test_energy_gradients_one_track():
    v = [np.array([[1.]]), np.array([[1.]])]
    # straight track
    pos = np.array([[0., 0], [1., 0], [2., 0]])
    seg = [np.array([[0], [1]]), np.array([[1], [2]])]
    eg = energy_gradients(pos, seg, drop_gradients_on_self=False)(v)
    assert [len(e) for e in eg] == [2, 2, 2]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], [[-0.5]])  # reward for straight track
    assert_array_almost_equal(eng[0], [[-1]])  # 2 active for 3 hits, wants more
    assert_array_almost_equal(efg[0], [[0]])  # no forks or joins possible
    assert_array_almost_equal(ecg[1], [[-0.5]])
    assert_array_almost_equal(eng[1], [[-1]])
    assert_array_almost_equal(efg[1], [[0]])

    # curved track
    pos = np.array([[0., 0], [1., 0], [2., 1]])
    eg = energy_gradients(pos, seg, drop_gradients_on_self=False)(v)
    assert [len(e) for e in eg] == [2, 2, 2]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], [[-1. / 8]])
    assert_array_almost_equal(eng[0], [[-1.]])
    assert_array_almost_equal(efg[0], [[0]])
    assert_array_almost_equal(ecg[1], [[-1. / 8]])
    assert_array_almost_equal(eng[1], [[-1.]])
    assert_array_almost_equal(efg[1], [[0]])

    # straight track
    pos = np.array([[0., 0], [1., 0], [2., 0]])
    eg = energy_gradients(pos, seg, drop_gradients_on_self=True)(v)
    assert [len(e) for e in eg] == [2, 2, 2]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], [[-0.5]])  # reward for straight track
    assert_array_almost_equal(eng[0], [[-2]])  # 2 - itself = 1 active for 3 hits, wants more
    assert_array_almost_equal(efg[0], [[0]])  # no forks or joins possible
    assert_array_almost_equal(ecg[1], [[-0.5]])
    assert_array_almost_equal(eng[1], [[-2]])
    assert_array_almost_equal(efg[1], [[0]])

    # curved track
    pos = np.array([[0., 0], [1., 0], [2., 1]])
    eg = energy_gradients(pos, seg, drop_gradients_on_self=True)(v)
    assert [len(e) for e in eg] == [2, 2, 2]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], [[-1. / 8]])
    assert_array_almost_equal(eng[0], [[-2]])
    assert_array_almost_equal(efg[0], [[0]])
    assert_array_almost_equal(ecg[1], [[-1. / 8]])
    assert_array_almost_equal(eng[1], [[-2]])
    assert_array_almost_equal(efg[1], [[0]])
