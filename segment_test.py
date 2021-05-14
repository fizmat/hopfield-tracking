from pytest import approx
from torch import tensor, zeros, ones

from segment import number_of_forks_energy, number_of_used_vertices_energy, curvature_energy, count_vertices, \
    count_segments


def test_number_of_forks_energy():
    none = tensor([[0, 0], [0, 0]])
    track = tensor([[1, 0], [0, 0]])
    parallel = tensor([[1, 0], [0, 1]])
    cross = tensor([[0, 1], [1, 0]])
    fork = tensor([[1, 1], [0, 0]])
    join = tensor([[0, 0], [1, 1]])
    zed = tensor([[1, 0], [1, 1]])
    full = tensor([[1, 1], [1, 1]])
    assert number_of_forks_energy(none) == 0
    assert number_of_forks_energy(track) == 0
    assert number_of_forks_energy(parallel) == 0
    assert number_of_forks_energy(cross) == 0
    assert number_of_forks_energy(fork) == 1.
    assert number_of_forks_energy(join) == 1.
    assert number_of_forks_energy(zed) == 2.
    assert number_of_forks_energy(full) == 4.
    assert number_of_forks_energy(0.5 * fork) == 0.25
    assert number_of_forks_energy(0.5 * join) == 0.25
    assert number_of_forks_energy(0.5 * zed) == 0.5
    assert number_of_forks_energy(0.5 * full) == 1.


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
