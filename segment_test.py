from pytest import approx
from torch import tensor, zeros, ones

from segment import track_crossing_energy, number_of_used_vertices_energy, curvature_energy, count_vertices, \
    count_segments, fork_energy, join_energy

none = tensor([[0, 0], [0, 0]])
track = tensor([[1, 0], [0, 0]])
parallel = tensor([[1, 0], [0, 1]])
cross = tensor([[0, 1], [1, 0]])
fork = tensor([[1, 1], [0, 0]])
join = tensor([[1, 0], [1, 0]])
zed = tensor([[1, 0], [1, 1]])
full = tensor([[1, 1], [1, 1]])


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


def test_number_of_used_vertices_energy():
    assert number_of_used_vertices_energy(0, 0) == 0
    assert number_of_used_vertices_energy(6, 0) == 18
    assert number_of_used_vertices_energy(6, 1) == 12.5
    assert number_of_used_vertices_energy(6, 2) == 8
    assert number_of_used_vertices_energy(6, 3) == 4.5
    assert number_of_used_vertices_energy(6, 4) == 2
    assert number_of_used_vertices_energy(6, 5) == 0.5
    assert number_of_used_vertices_energy(6, 6) == 0
    assert number_of_used_vertices_energy(6, 7) == 0.5
    assert number_of_used_vertices_energy(6, 9) == 4.5


def test_curvature_energy():
    a = tensor([[0., 0]])
    b = tensor([[1., 0]])
    c = tensor([[2., 0], [2, 1], [2, -1], [3, 0], [3, 2]])
    f = curvature_energy(a, b, c)
    first = tensor([[1]])
    assert f(first, tensor([[1, 0, 0, 0, 0]])) == - 0.5
    assert f(first, tensor([[0, 1, 0, 0, 0]])) == approx(- 1. / 8)
    assert f(first, tensor([[0, 0, 1, 0, 0]])) == approx(- 1. / 8)
    assert f(first, tensor([[0, 0, 0, 1, 0]])) == approx(- 1. / 4)
    assert f(first, tensor([[0, 0, 0, 0, 1]])) == approx(- 1. / 16)
    assert f(first, tensor([[1, 1, 1, 1, 1]])) == approx(- 17. / 16)
    assert f(first, tensor([[.1, .1, .1, .1, .1]])) == approx(- 1.7 / 16)
