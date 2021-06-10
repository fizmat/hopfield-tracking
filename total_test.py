import numpy as np
from numpy.testing import assert_array_equal

from total import total_activation_matrix_, number_of_used_vertices_energy, \
    number_of_used_vertices_energy_gradient, total_activation_matrix


def test_total_activation_matrix_():
    a, b, c = total_activation_matrix_(0, 0)
    assert a.shape == (0, 0)
    assert b.shape == (0,)
    assert c == 0
    a, b, c = total_activation_matrix_(6, 3)
    assert_array_equal(a, np.full((3, 3), 0.5))
    assert_array_equal(b, np.full(3, -6))
    assert c == 18


def test_total_activation_matrix():
    a, b, c = total_activation_matrix(np.empty(0), [])
    assert a.shape == (0, 0)
    assert b.shape == (0,)
    assert c == 0
    a, b, c = total_activation_matrix(np.empty(6), [np.empty(3)])
    assert_array_equal(a, np.full((3, 3), 0.5))
    assert_array_equal(b, np.full(3, -6))
    assert c == 18


def test_number_of_used_vertices_energy():
    a, b, c = total_activation_matrix_(0, 0)
    assert number_of_used_vertices_energy(a, b, c, np.array([])) == 0
    a, b, c = total_activation_matrix_(6, 1)
    assert number_of_used_vertices_energy(a, b, c, np.array([0])) == 18
    assert number_of_used_vertices_energy(a, b, c, np.array([1])) == 12.5
    assert number_of_used_vertices_energy(a, b, c, np.array([2])) == 8
    assert number_of_used_vertices_energy(a, b, c, np.array([3])) == 4.5
    assert number_of_used_vertices_energy(a, b, c, np.array([4])) == 2
    assert number_of_used_vertices_energy(a, b, c, np.array([5])) == 0.5
    assert number_of_used_vertices_energy(a, b, c, np.array([6])) == 0
    assert number_of_used_vertices_energy(a, b, c, np.array([7])) == 0.5
    assert number_of_used_vertices_energy(a, b, c, np.array([9])) == 4.5


def test_count_vertices_gradient():
    assert number_of_used_vertices_energy_gradient(0, 0) == 0
    assert number_of_used_vertices_energy_gradient(6, 0) == -6
    assert number_of_used_vertices_energy_gradient(6, 2.) == -4
    assert number_of_used_vertices_energy_gradient(6, 6.) == 0
    assert number_of_used_vertices_energy_gradient(6, 8.) == 2
