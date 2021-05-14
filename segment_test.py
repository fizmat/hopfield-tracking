from segment import number_of_forks_energy, number_of_used_vertices_energy
from torch import tensor


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


def test_number_of_used_vertices_energy():
    f = number_of_used_vertices_energy(range(2), range(2), range(2))
    none = tensor([[0, 0], [0, 0]])
    track = tensor([[1, 0], [0, 0]])
    parallel = tensor([[1, 0], [0, 1]])
    cross = tensor([[0, 1], [1, 0]])
    fork = tensor([[1, 1], [0, 0]])
    join = tensor([[0, 0], [1, 1]])
    zed = tensor([[1, 0], [1, 1]])
    full = tensor([[1, 1], [1, 1]])
    assert f(none, none) == 18
    assert f(track, track) == 8
    assert f(parallel, parallel) == 2
    assert f(cross, cross) == 2
    assert f(fork, fork) == 2
    assert f(join, join) == 2
    assert f(zed, zed) == 0
    assert f(full, full) == 2
    assert f(0.5 * track, 0.5 * track) == 12.5
    assert f(0.5 * join, 0.5 * join) == 8
    assert f(0.5 * full, 0.5 * full) == 2
