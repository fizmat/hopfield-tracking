import numpy as np
import torch
import torch_testing as tt
from numpy.testing import assert_array_almost_equal_nulp, assert_array_almost_equal
from pytest import approx
from torch import tensor, zeros, ones

from segment import track_crossing_energy, number_of_used_vertices_energy, curvature_energy, count_vertices, \
    count_segments, fork_energy, join_energy, energy, curvature_energy_matrix, fork_energy_gradient, \
    join_energy_gradient, curvature_energy_gradient, energy_gradients, number_of_used_vertices_energy_gradient

none = tensor([[0., 0], [0, 0]])
track = tensor([[1., 0], [0, 0]])
parallel = tensor([[1., 0], [0, 1]])
cross = tensor([[0., 1], [1, 0]])
fork = tensor([[1., 1], [0, 0]])
join = tensor([[1., 0], [1, 0]])
zed = tensor([[1., 0], [1, 1]])
full = tensor([[1., 1], [1, 1]])


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


def test_fork_energy_grad():
    def grad(v):
        _v = v.clone().detach().requires_grad_(True)
        fork_energy(_v).backward()
        return _v.grad

    tt.assert_equal(grad(none), zeros(2, 2))
    tt.assert_equal(grad(track), tensor([[0, 1], [0, 0]]))
    tt.assert_equal(grad(parallel), tensor([[0, 1], [1, 0]]))
    tt.assert_equal(grad(cross), tensor([[1, 0], [0, 1]]))
    tt.assert_equal(grad(fork), tensor([[1, 1], [0, 0]]))
    tt.assert_equal(grad(join), tensor([[0, 1], [0, 1]]))
    tt.assert_equal(grad(zed), tensor([[0, 1], [1, 1]]))
    tt.assert_equal(grad(full), tensor([[1, 1], [1, 1]]))
    tt.assert_equal(grad(0.5 * fork), tensor([[.5, .5], [0, 0]]))
    tt.assert_equal(grad(0.5 * join), tensor([[0, .5], [0, .5]]))
    tt.assert_equal(grad(0.5 * zed), tensor([[0, .5], [.5, .5]]))
    tt.assert_equal(grad(0.5 * full), tensor([[.5, .5], [.5, .5]]))


def test_fork_energy_gradient():
    grad = fork_energy_gradient
    tt.assert_equal(grad(none), zeros(2, 2))
    tt.assert_equal(grad(track), tensor([[0, 1], [0, 0]]))
    tt.assert_equal(grad(parallel), tensor([[0, 1], [1, 0]]))
    tt.assert_equal(grad(cross), tensor([[1, 0], [0, 1]]))
    tt.assert_equal(grad(fork), tensor([[1, 1], [0, 0]]))
    tt.assert_equal(grad(join), tensor([[0, 1], [0, 1]]))
    tt.assert_equal(grad(zed), tensor([[0, 1], [1, 1]]))
    tt.assert_equal(grad(full), tensor([[1, 1], [1, 1]]))
    tt.assert_equal(grad(0.5 * fork), tensor([[.5, .5], [0, 0]]))
    tt.assert_equal(grad(0.5 * join), tensor([[0, .5], [0, .5]]))
    tt.assert_equal(grad(0.5 * zed), tensor([[0, .5], [.5, .5]]))
    tt.assert_equal(grad(0.5 * full), tensor([[.5, .5], [.5, .5]]))


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
    def grad(v):
        _v = v.clone().detach().requires_grad_(True)
        join_energy(_v).backward()
        return _v.grad

    tt.assert_equal(grad(none), zeros(2, 2))
    tt.assert_equal(grad(track), tensor([[0, 0], [1, 0]]))
    tt.assert_equal(grad(parallel), tensor([[0, 1], [1, 0]]))
    tt.assert_equal(grad(cross), tensor([[1, 0], [0, 1]]))
    tt.assert_equal(grad(fork), tensor([[0, 0], [1, 1]]))
    tt.assert_equal(grad(join), tensor([[1, 0], [1, 0]]))
    tt.assert_equal(grad(zed), tensor([[1, 1], [1, 0]]))
    tt.assert_equal(grad(full), tensor([[1, 1], [1, 1]]))
    tt.assert_equal(grad(0.5 * fork), tensor([[0, 0], [.5, .5]]))
    tt.assert_equal(grad(0.5 * join), tensor([[.5, 0], [.5, 0]]))
    tt.assert_equal(grad(0.5 * zed), tensor([[.5, .5], [.5, 0]]))
    tt.assert_equal(grad(0.5 * full), tensor([[.5, .5], [.5, .5]]))


def test_join_energy_grad():
    grad = join_energy_gradient
    tt.assert_equal(grad(none), zeros(2, 2))
    tt.assert_equal(grad(track), tensor([[0, 0], [1, 0]]))
    tt.assert_equal(grad(parallel), tensor([[0, 1], [1, 0]]))
    tt.assert_equal(grad(cross), tensor([[1, 0], [0, 1]]))
    tt.assert_equal(grad(fork), tensor([[0, 0], [1, 1]]))
    tt.assert_equal(grad(join), tensor([[1, 0], [1, 0]]))
    tt.assert_equal(grad(zed), tensor([[1, 1], [1, 0]]))
    tt.assert_equal(grad(full), tensor([[1, 1], [1, 1]]))
    tt.assert_equal(grad(0.5 * fork), tensor([[0, 0], [.5, .5]]))
    tt.assert_equal(grad(0.5 * join), tensor([[.5, 0], [.5, 0]]))
    tt.assert_equal(grad(0.5 * zed), tensor([[.5, .5], [.5, 0]]))
    tt.assert_equal(grad(0.5 * full), tensor([[.5, .5], [.5, .5]]))


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


def test_track_crossing_energy_grad():
    def grad(v):
        _v = v.clone().detach().requires_grad_(True)
        track_crossing_energy(_v).backward()
        return _v.grad

    tt.assert_equal(grad(zed), tensor([[1, 2], [2, 1]]))


def test_count_vertices():
    assert count_vertices(()) == 0
    assert count_vertices([zeros(2, 4)]) == 2
    assert count_vertices([ones(2, 2), ones(3, 2)]) == 5


def test_count_segments():
    assert count_segments([]) == 0
    assert count_segments([zeros(2, 4)]) == 0
    assert count_segments([ones(3, 2), ones(2, 4)]) == 14
    assert count_segments((
        tensor([[0.5, 0.2], [0, 0.1]]),
        tensor([[0.1], [0.2]])
    )) == approx(1.1)


def test_count_segments_grad():
    v = (
        tensor([[0.5, 0.2], [0, 0.1]], requires_grad=True),
        tensor([[0.1], [0.2]], requires_grad=True)
    )
    count_segments(v).backward()
    tt.assert_equal(v[0].grad, ones(2, 2))
    tt.assert_equal(v[1].grad, ones(2, 1))


def test_number_of_used_vertices_energy():
    assert number_of_used_vertices_energy(0, tensor(0)) == 0
    assert number_of_used_vertices_energy(6, tensor(0)) == 18
    assert number_of_used_vertices_energy(6, tensor(1)) == 12.5
    assert number_of_used_vertices_energy(6, tensor(2)) == 8
    assert number_of_used_vertices_energy(6, tensor(3)) == 4.5
    assert number_of_used_vertices_energy(6, tensor(4)) == 2
    assert number_of_used_vertices_energy(6, tensor(5)) == 0.5
    assert number_of_used_vertices_energy(6, tensor(6)) == 0
    assert number_of_used_vertices_energy(6, tensor(7)) == 0.5
    assert number_of_used_vertices_energy(6, tensor(9)) == 4.5


def test_count_vertices_grad():
    v = tensor(0., requires_grad=True)
    number_of_used_vertices_energy(0, v).backward()
    assert v.grad == 0
    v = tensor(0., requires_grad=True)
    number_of_used_vertices_energy(6, v).backward()
    assert v.grad == -6
    v = tensor(2., requires_grad=True)
    number_of_used_vertices_energy(6, v).backward()
    assert v.grad == -4
    v = tensor(6., requires_grad=True)
    number_of_used_vertices_energy(6, v).backward()
    assert v.grad == 0
    v = tensor(8., requires_grad=True)
    number_of_used_vertices_energy(6, v).backward()
    assert v.grad == 2


def test_count_vertices_gradient():
    assert number_of_used_vertices_energy_gradient(0, 0) == 0
    assert number_of_used_vertices_energy_gradient(6, 0) == -6
    assert number_of_used_vertices_energy_gradient(6, 2.) == -4
    assert number_of_used_vertices_energy_gradient(6, 6.) == 0
    assert number_of_used_vertices_energy_gradient(6, 8.) == 2


def test_curvature_energy_matrix():
    a = np.array([[0., 0]])
    b = np.array([[1., 0]])
    c = np.array([[2., 0], [2, 1], [2, -1], [3, 0], [3, 2]])
    w = curvature_energy_matrix(a, b, c)
    assert_array_almost_equal(w, [[[-1. / 2, -1. / 8, -1. / 8, -1. / 4, -1. / 16]]])


def test_curvature_energy():
    a = tensor([[0., 0]])
    b = tensor([[1., 0]])
    c = tensor([[2., 0], [2, 1], [2, -1], [3, 0], [3, 2]])
    w = tensor(curvature_energy_matrix(a, b, c))
    first = tensor([[1]])
    assert curvature_energy(w, first, tensor([[1, 0, 0, 0, 0]])) == - 0.5
    assert curvature_energy(w, first, tensor([[0, 1, 0, 0, 0]])) == approx(- 1. / 8)
    assert curvature_energy(w, first, tensor([[0, 0, 1, 0, 0]])) == approx(- 1. / 8)
    assert curvature_energy(w, first, tensor([[0, 0, 0, 1, 0]])) == approx(- 1. / 4)
    assert curvature_energy(w, first, tensor([[0, 0, 0, 0, 1]])) == approx(- 1. / 16)
    assert curvature_energy(w, first, tensor([[1, 1, 1, 1, 1]])) == approx(- 17. / 16)
    assert curvature_energy(w, first, tensor([[.1, .1, .1, .1, .1]])) == approx(- 1.7 / 16)


def test_curvature_energy_grad():
    a = tensor([[0., 0]])
    b = tensor([[1., 0]])
    c = tensor([[2., 0], [2, 1], [2, -1], [3, 0], [3, 2]])
    w = tensor(curvature_energy_matrix(a, b, c))
    first = tensor([[1.]], requires_grad=True)
    second = tensor([[1., 0, 0, 0, 0]], requires_grad=True)
    curvature_energy(w, first, second).backward()
    assert first.grad == - 0.5
    tt.assert_almost_equal(second.grad, tensor([[- 0.5, -1. / 8, -1. / 8, -1. / 4, -1. / 16]]))


def test_curvature_energy_gradient():
    a = tensor([[0., 0]])
    b = tensor([[1., 0]])
    c = tensor([[2., 0], [2, 1], [2, -1], [3, 0], [3, 2]])
    w = tensor(curvature_energy_matrix(a, b, c))
    first = tensor([[1.]])
    second = tensor([[1., 0, 0, 0, 0]])
    g1, g2 = curvature_energy_gradient(w, first, second)
    tt.assert_almost_equal(g1, tensor([[-0.5]]))
    tt.assert_almost_equal(g2, tensor([[- 0.5, -1. / 8, -1. / 8, -1. / 4, -1. / 16]]))


def test_energy_empty():
    assert energy([])([]) == 0


def test_energy_gradients_empty():
    assert energy_gradients([])([]) == ([], [], [])


def test_energy_two_hits():
    # vertex count energy only
    pos = [tensor([[0., 0]]),
           tensor([[1, 0]])]
    v = tensor([[0.]], requires_grad=True)
    assert energy(pos)([v]) == 2
    energy(pos)([v]).backward()
    assert v.grad == -2
    v = tensor([[1.]], requires_grad=True)
    assert energy(pos)([v]) == 0.5
    energy(pos)([v]).backward()
    assert v.grad == -1


def test_energy_gradients_two_hits():
    # vertex count energy only
    pos = [np.array([[0., 0]]),
           np.array([[1, 0]])]
    v = np.array([[0.]])
    eg = energy_gradients(pos)([v])
    assert [len(e) for e in eg] == [1, 1, 1]
    ecg, eng, efg = eg
    assert_array_almost_equal_nulp(ecg[0], [[0]])
    assert_array_almost_equal_nulp(eng[0], [[-2]])
    assert_array_almost_equal_nulp(efg[0], [[0]])
    v = np.array([[1.]])
    eg = energy_gradients(pos)([v])
    assert [len(e) for e in eg] == [1, 1, 1]
    ecg, eng, efg = eg
    assert_array_almost_equal_nulp(ecg[0], [[0]])
    assert_array_almost_equal_nulp(eng[0], [[-1]])
    assert_array_almost_equal_nulp(efg[0], [[0]])


def test_energy_four_hits():
    pos = [tensor([[0., 0], [0, 1]]),
           tensor([[1, 0], [1, 1]])]
    e = energy(pos)
    v = zeros(2, 2, requires_grad=True)
    assert e([v]) == 8  # bad vertex count
    e([v]).backward()
    tt.assert_almost_equal(v.grad, torch.full((2, 2), -4.))
    v = ones(2, 2, requires_grad=True)
    assert e([v]) == 4  # many forks
    e([v]).backward()
    tt.assert_almost_equal(v.grad, torch.full((2, 2), 2.))


def test_energy_gradients_four_hits():
    # vertex count energy only
    pos = [np.array([[0., 0], [0, 1]]),
           np.array([[1, 0], [1, 1]])]
    v = np.zeros((2, 2))
    eg = energy_gradients(pos)([v])
    assert [len(e) for e in eg] == [1, 1, 1]
    ecg, eng, efg = eg
    assert_array_almost_equal_nulp(ecg[0], np.zeros((2, 2)))
    assert_array_almost_equal_nulp(eng[0], np.full((2, 2), -4))
    assert_array_almost_equal_nulp(efg[0], np.zeros((2, 2)))
    v = np.ones((2, 2))
    eg = energy_gradients(pos)([v])
    assert [len(e) for e in eg] == [1, 1, 1]
    ecg, eng, efg = eg
    assert_array_almost_equal_nulp(ecg[0], np.zeros((2, 2)))  # short tracks, no cosines yet
    assert_array_almost_equal_nulp(eng[0], np.zeros((2, 2)))  # exactly 4 active for 4 vertices
    assert_array_almost_equal_nulp(efg[0], np.full((2, 2), 2))  # forks and joins


def test_energy_one_track():
    v = [tensor([[1.]], requires_grad=True), tensor([[1.]], requires_grad=True)]
    e = energy([tensor([[0., 0]]), tensor([[1., 0]]), tensor([[2., 0]])])
    assert e(v) == 0.5 + 0 - 0.5
    e(v).backward()
    assert v[0].grad == -1 + 0 - 0.5
    assert v[1].grad == -1 + 0 - 0.5

    v = [tensor([[1.]], requires_grad=True), tensor([[1.]], requires_grad=True)]
    e = energy([tensor([[0., 0]]), tensor([[1., 0]]), tensor([[2., 1]])])
    assert e(v).item() == approx(0.5 + 0 - 1. / 8)
    e(v).backward()
    assert v[0].grad == -1 + 0 - 1. / 8
    assert v[1].grad == -1 + 0 - 1. / 8


def test_energy_gradients_one_track():
    v = [np.array([[1.]]), np.array([[1.]])]
    # straight track
    pos = [np.array([[0., 0]]), np.array([[1., 0]]), np.array([[2., 0]])]
    eg = energy_gradients(pos)(v)
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
    eg = energy_gradients(pos)(v)
    assert [len(e) for e in eg] == [2, 2, 2]
    ecg, eng, efg = eg
    assert_array_almost_equal(ecg[0], [[-1. / 8]])
    assert_array_almost_equal(eng[0], [[-1.]])
    assert_array_almost_equal(efg[0], [[0]])
    assert_array_almost_equal(ecg[1], [[-1. / 8]])
    assert_array_almost_equal(eng[1], [[-1.]])
    assert_array_almost_equal(efg[1], [[0]])
