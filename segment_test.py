from pytest import approx

from segment import number_of_forks_energy, number_of_used_vertices_energy, curvature_energy
from torch import tensor
import numpy as np


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
