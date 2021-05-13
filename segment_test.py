from segment import number_of_forks_energy
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
