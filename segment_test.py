import numpy as np
from numpy.testing import assert_array_almost_equal
from pytest import approx

from segment import track_crossing_energy, number_of_used_vertices_energy, curvature_energy, count_vertices, \
    count_segments, fork_energy, join_energy, energy, curvature_energy_matrix, fork_energy_gradient, \
    join_energy_gradient, curvature_energy_gradient, energy_gradients, number_of_used_vertices_energy_gradient, \
    curvature_energy_pairwise

none = np.array([[0., 0], [0, 0]])
track = np.array([[1., 0], [0, 0]])
parallel = np.array([[1., 0], [0, 1]])
cross = np.array([[0., 1], [1, 0]])
fork = np.array([[1., 1], [0, 0]])
join = np.array([[1., 0], [1, 0]])
zed = np.array([[1., 0], [1, 1]])
full = np.array([[1., 1], [1, 1]])


def test_fork_energy():
    assert fork_energy(none) == 0
    assert fork_energy(track) == 0
    assert fork_energy(parallel) == 0
    assert fork_energy(cross) == 0
    assert fork_energy(fork) == 1.
    assert fork_energy(join) == 0.
    assert fork_energy(zed) == 1.
    assert fork_energy(full) == 2.
    assert fork_energy(0.5 * fork) == 0.25
    assert fork_energy(0.5 * join) == 0
    assert fork_energy(0.5 * zed) == 0.25
    assert fork_energy(0.5 * full) == 0.5


def test_fork_energy_gradient():
    grad = fork_energy_gradient
    assert_array_almost_equal(grad(none), np.zeros((2, 2)))
    assert_array_almost_equal(grad(track), np.array([[0, 1], [0, 0]]))
    assert_array_almost_equal(grad(parallel), np.array([[0, 1], [1, 0]]))
    assert_array_almost_equal(grad(cross), np.array([[1, 0], [0, 1]]))
    assert_array_almost_equal(grad(fork), np.array([[1, 1], [0, 0]]))
    assert_array_almost_equal(grad(join), np.array([[0, 1], [0, 1]]))
    assert_array_almost_equal(grad(zed), np.array([[0, 1], [1, 1]]))
    assert_array_almost_equal(grad(full), np.array([[1, 1], [1, 1]]))
    assert_array_almost_equal(grad(0.5 * fork), np.array([[.5, .5], [0, 0]]))
    assert_array_almost_equal(grad(0.5 * join), np.array([[0, .5], [0, .5]]))
    assert_array_almost_equal(grad(0.5 * zed), np.array([[0, .5], [.5, .5]]))
    assert_array_almost_equal(grad(0.5 * full), np.array([[.5, .5], [.5, .5]]))


def test_join_energy():
    assert join_energy(none) == 0
    assert join_energy(track) == 0
    assert join_energy(parallel) == 0
    assert join_energy(cross) == 0
    assert join_energy(fork) == 0
    assert join_energy(join) == 1.
    assert join_energy(zed) == 1.
    assert join_energy(full) == 2.
    assert join_energy(0.5 * fork) == 0
    assert join_energy(0.5 * join) == 0.25
    assert join_energy(0.5 * zed) == 0.25
    assert join_energy(0.5 * full) == 0.5


def test_join_energy_grad():
    grad = join_energy_gradient
    assert_array_almost_equal(grad(none), np.zeros((2, 2)))
    assert_array_almost_equal(grad(track), np.array([[0, 0], [1, 0]]))
    assert_array_almost_equal(grad(parallel), np.array([[0, 1], [1, 0]]))
    assert_array_almost_equal(grad(cross), np.array([[1, 0], [0, 1]]))
    assert_array_almost_equal(grad(fork), np.array([[0, 0], [1, 1]]))
    assert_array_almost_equal(grad(join), np.array([[1, 0], [1, 0]]))
    assert_array_almost_equal(grad(zed), np.array([[1, 1], [1, 0]]))
    assert_array_almost_equal(grad(full), np.array([[1, 1], [1, 1]]))
    assert_array_almost_equal(grad(0.5 * fork), np.array([[0, 0], [.5, .5]]))
    assert_array_almost_equal(grad(0.5 * join), np.array([[.5, 0], [.5, 0]]))
    assert_array_almost_equal(grad(0.5 * zed), np.array([[.5, .5], [.5, 0]]))
    assert_array_almost_equal(grad(0.5 * full), np.array([[.5, .5], [.5, .5]]))


def test_track_crossing_energy():
    assert track_crossing_energy(none) == 0
    assert track_crossing_energy(track) == 0
    assert track_crossing_energy(parallel) == 0
    assert track_crossing_energy(cross) == 0
    assert track_crossing_energy(fork) == 1.
    assert track_crossing_energy(join) == 1.
    assert track_crossing_energy(zed) == 2.
    assert track_crossing_energy(full) == 4.
    assert track_crossing_energy(0.5 * fork) == 0.25
    assert track_crossing_energy(0.5 * join) == 0.25
    assert track_crossing_energy(0.5 * zed) == 0.5
    assert track_crossing_energy(0.5 * full) == 1.


def test_count_vertices():
    assert count_vertices(()) == 0
    assert count_vertices([np.zeros((2, 4))]) == 2
    assert count_vertices([np.ones((2, 2)), np.ones((3, 2))]) == 5


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
    a = np.array([[0., 0]])
    b = np.array([[1., 0]])
    c = np.array([[2., 0], [2, 1], [2, -1], [3, 0], [3, 2]])
    w = curvature_energy_matrix(a, b, c)
    assert_array_almost_equal(w, [[[-1. / 2, -1. / 8, -1. / 8, -1. / 4, -1. / 16]]])


def test_curvature_energy():
    a = np.array([[0., 0]])
    b = np.array([[1., 0]])
    c = np.array([[2., 0], [2, 1], [2, -1], [3, 0], [3, 2]])
    w = curvature_energy_matrix(a, b, c)
    first = np.array([[1]])
    assert curvature_energy(w, first, np.array([[1, 0, 0, 0, 0]])) == - 0.5
    assert curvature_energy(w, first, np.array([[0, 1, 0, 0, 0]])) == approx(- 1. / 8)
    assert curvature_energy(w, first, np.array([[0, 0, 1, 0, 0]])) == approx(- 1. / 8)
    assert curvature_energy(w, first, np.array([[0, 0, 0, 1, 0]])) == approx(- 1. / 4)
    assert curvature_energy(w, first, np.array([[0, 0, 0, 0, 1]])) == approx(- 1. / 16)
    assert curvature_energy(w, first, np.array([[1, 1, 1, 1, 1]])) == approx(- 17. / 16)
    assert curvature_energy(w, first, np.array([[.1, .1, .1, .1, .1]])) == approx(- 1.7 / 16)


def test_curvature_energy_gradient():
    a = np.array([[0., 0]])
    b = np.array([[1., 0]])
    c = np.array([[2., 0], [2, 1], [2, -1], [3, 0], [3, 2]])
    w = curvature_energy_matrix(a, b, c)
    first = np.array([[1.]])
    second = np.array([[1., 0, 0, 0, 0]])
    g1, g2 = curvature_energy_gradient(w, first, second)
    assert_array_almost_equal(g1, np.array([[-0.5]]))
    assert_array_almost_equal(g2, np.array([[- 0.5, -1. / 8, -1. / 8, -1. / 4, -1. / 16]]))


def test_energy_empty():
    assert energy([])([]) == 0


def test_energy_gradients_empty():
    assert energy_gradients([])([]) == ([], [], [])


def test_energy_two_hits():
    # vertex count energy only
    pos = [np.array([[0., 0]]),
           np.array([[1, 0]])]
    v = np.array([[0.]])
    assert energy(pos)([v]) == 2
    v = np.array([[1.]])
    assert energy(pos)([v]) == 0.5


def test_energy_gradients_two_hits():
    # vertex count energy only
    pos = [np.array([[0., 0]]),
           np.array([[1, 0]])]
    v = np.array([[0.]])
    eg = energy_gradients(pos, drop_gradients_on_self=False)([v])
    assert [len(e) for e in eg] == [1, 1, 1]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], [[0]])
    assert_array_almost_equal(eng[0], [[-2]])
    assert_array_almost_equal(efg[0], [[0]])
    v = np.array([[1.]])
    eg = energy_gradients(pos, drop_gradients_on_self=False)([v])
    assert [len(e) for e in eg] == [1, 1, 1]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], [[0]])
    assert_array_almost_equal(eng[0], [[-1]])
    assert_array_almost_equal(efg[0], [[0]])

    eg = energy_gradients(pos, drop_gradients_on_self=True)([v])
    assert [len(e) for e in eg] == [1, 1, 1]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], [[0]])
    assert_array_almost_equal(eng[0], [[-2]])
    assert_array_almost_equal(efg[0], [[0]])
    v = np.array([[1.]])
    eg = energy_gradients(pos, drop_gradients_on_self=True)([v])
    assert [len(e) for e in eg] == [1, 1, 1]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], [[0]])
    assert_array_almost_equal(eng[0], [[-2]])
    assert_array_almost_equal(efg[0], [[0]])


def test_energy_four_hits():
    pos = [np.array([[0., 0], [0, 1]]),
           np.array([[1, 0], [1, 1]])]
    e = energy(pos)
    v = np.zeros((2, 2))
    assert e([v]) == 8  # bad vertex count
    v = np.ones((2, 2))
    assert e([v]) == 4  # many forks


def test_energy_gradients_four_hits():
    # vertex count energy only
    pos = [np.array([[0., 0], [0, 1]]),
           np.array([[1, 0], [1, 1]])]
    v = np.zeros((2, 2))
    eg = energy_gradients(pos, drop_gradients_on_self=False)([v])
    assert [len(e) for e in eg] == [1, 1, 1]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], np.zeros((2, 2)))
    assert_array_almost_equal(eng[0], np.full((2, 2), -4))
    assert_array_almost_equal(efg[0], np.zeros((2, 2)))
    v = np.ones((2, 2))
    eg = energy_gradients(pos, drop_gradients_on_self=False)([v])
    assert [len(e) for e in eg] == [1, 1, 1]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], np.zeros((2, 2)))  # short tracks, no cosines yet
    assert_array_almost_equal(eng[0], np.zeros((2, 2)))  # exactly 4 active for 4 vertices
    assert_array_almost_equal(efg[0], np.full((2, 2), 2))  # forks and joins

    v = np.zeros((2, 2))
    eg = energy_gradients(pos, drop_gradients_on_self=True)([v])
    assert [len(e) for e in eg] == [1, 1, 1]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], np.zeros((2, 2)))
    assert_array_almost_equal(eng[0], np.full((2, 2), -4))
    assert_array_almost_equal(efg[0], np.zeros((2, 2)))
    v = np.ones((2, 2))
    eg = energy_gradients(pos, drop_gradients_on_self=True)([v])
    assert [len(e) for e in eg] == [1, 1, 1]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], np.zeros((2, 2)))  # short tracks, no cosines yet
    assert_array_almost_equal(eng[0], np.full((2, 2), -1))  # 4 active for 4 vertices
    assert_array_almost_equal(efg[0], np.full((2, 2), 2))  # forks and joins


def test_energy_one_track():
    v = [np.array([[1.]]), np.array([[1.]])]
    e = energy([np.array([[0., 0]]), np.array([[1., 0]]), np.array([[2., 0]])])
    assert e(v) == 0.5 + 0 - 0.5

    v = [np.array([[1.]]), np.array([[1.]])]
    e = energy([np.array([[0., 0]]), np.array([[1., 0]]), np.array([[2., 1]])])
    assert e(v).item() == approx(0.5 + 0 - 1. / 8)


def test_energy_gradients_one_track():
    v = [np.array([[1.]]), np.array([[1.]])]
    # straight track
    pos = [np.array([[0., 0]]), np.array([[1., 0]]), np.array([[2., 0]])]
    eg = energy_gradients(pos, drop_gradients_on_self=False)(v)
    assert [len(e) for e in eg] == [2, 2, 2]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], [[-0.5]])  # reward for straight track
    assert_array_almost_equal(eng[0], [[-1]])  # 2 active for 3 hits, wants more
    assert_array_almost_equal(efg[0], [[0]])  # no forks or joins possible
    assert_array_almost_equal(ecg[1], [[-0.5]])
    assert_array_almost_equal(eng[1], [[-1]])
    assert_array_almost_equal(efg[1], [[0]])

    # curved track
    pos = [np.array([[0., 0]]), np.array([[1., 0]]), np.array([[2., 1]])]
    eg = energy_gradients(pos, drop_gradients_on_self=False)(v)
    assert [len(e) for e in eg] == [2, 2, 2]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], [[-1. / 8]])
    assert_array_almost_equal(eng[0], [[-1.]])
    assert_array_almost_equal(efg[0], [[0]])
    assert_array_almost_equal(ecg[1], [[-1. / 8]])
    assert_array_almost_equal(eng[1], [[-1.]])
    assert_array_almost_equal(efg[1], [[0]])

    # straight track
    pos = [np.array([[0., 0]]), np.array([[1., 0]]), np.array([[2., 0]])]
    eg = energy_gradients(pos, drop_gradients_on_self=True)(v)
    assert [len(e) for e in eg] == [2, 2, 2]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], [[-0.5]])  # reward for straight track
    assert_array_almost_equal(eng[0], [[-2]])  # 2 - itself = 1 active for 3 hits, wants more
    assert_array_almost_equal(efg[0], [[0]])  # no forks or joins possible
    assert_array_almost_equal(ecg[1], [[-0.5]])
    assert_array_almost_equal(eng[1], [[-2]])
    assert_array_almost_equal(efg[1], [[0]])

    # curved track
    pos = [np.array([[0., 0]]), np.array([[1., 0]]), np.array([[2., 1]])]
    eg = energy_gradients(pos, drop_gradients_on_self=True)(v)
    assert [len(e) for e in eg] == [2, 2, 2]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], [[-1. / 8]])
    assert_array_almost_equal(eng[0], [[-2]])
    assert_array_almost_equal(efg[0], [[0]])
    assert_array_almost_equal(ecg[1], [[-1. / 8]])
    assert_array_almost_equal(eng[1], [[-2]])
    assert_array_almost_equal(efg[1], [[0]])
