from segment import number_of_forks_energy
from torch import tensor


def test_number_of_forks_energy():
    none = tensor([[0, 0], [0, 0]])
    track = tensor([[1, 0], [0, 0]])
    parallel = tensor([[1, 0], [0, 1]])
    cross = tensor([[0, 1], [1, 0]])
    fork = tensor([[1, 1], [0, 0]])
    join = tensor([[0, 0], [1, 1]])
    full = tensor([[1, 1], [1, 1]])
    assert number_of_forks_energy(None, None, None)(none, none) == 0
    assert number_of_forks_energy(None, None, None)(track, track) == 0
    assert number_of_forks_energy(None, None, None)(parallel, parallel) == 0
    assert number_of_forks_energy(None, None, None)(cross, cross) == 0
    assert number_of_forks_energy(None, None, None)(fork, fork) == 2.
    assert number_of_forks_energy(None, None, None)(join, join) == 2.
    assert number_of_forks_energy(None, None, None)(full, full) == 8.
    assert number_of_forks_energy(None, None, None)(fork, none) == 1.
