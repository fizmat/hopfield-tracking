import numpy as np
from numpy.testing import assert_array_equal

from total import total_activation_matrix_, total_activation_energy, \
    total_activation_energy_gradient, total_activation_matrix


def test_total_activation_matrix_():
    a, b, c = total_activation_matrix_(0, 0)
    assert (a.shape, b.shape, c) == ((0, 0), (0,), 0)
    a, b, c = total_activation_matrix_(3, 0)
    assert (a.shape, b.shape, c) == ((0, 0), (0,), 4.5)
    a, b, c = total_activation_matrix_(2, 1)
    assert_array_equal(a, [[0]])
    assert_array_equal(b, [-2])
    assert c == 2
    a, b, c = total_activation_matrix_(6, 3)
    assert_array_equal(a, [[0, .5, .5],
                           [.5, 0, .5],
                           [.5, .5, 0]])
    assert_array_equal(b, [-6, -6, -6])
    assert c == 18

    a, b, c = total_activation_matrix_(0, 0, False)
    assert (a.shape, b.shape, c) == ((0, 0), (0,), 0)
    a, b, c = total_activation_matrix_(3, 0, False)
    assert (a.shape, b.shape, c) == ((0, 0), (0,), 4.5)
    a, b, c = total_activation_matrix_(2, 1, False)
    assert_array_equal(a, [[.5]])
    assert_array_equal(b, [-2])
    assert c == 2
    a, b, c = total_activation_matrix_(6, 3, False)
    assert_array_equal(a, [[.5, .5, .5],
                           [.5, .5, .5],
                           [.5, .5, .5]])
    assert_array_equal(b, [-6, -6, -6])
    assert c == 18


def test_total_activation_matrix():
    a, b, c = total_activation_matrix(np.empty(0), [])
    assert (a.shape, b.shape, c) == ((0, 0), (0,), 0)
    a, b, c = total_activation_matrix(np.empty((3, 3)), [])
    assert (a.shape, b.shape, c) == ((0, 0), (0,), 4.5)
    a, b, c = total_activation_matrix(np.empty((2, 3)), [np.array([[0, 1]])])
    assert_array_equal(a, [[0]])
    assert_array_equal(b, [-2])
    assert c == 2
    a, b, c = total_activation_matrix(np.empty(6), [np.empty(3)])
    assert_array_equal(a, [[0, .5, .5],
                           [.5, 0, .5],
                           [.5, .5, 0]])
    assert_array_equal(b, [-6, -6, -6])
    assert c == 18

    a, b, c = total_activation_matrix(np.empty(0), [], False)
    assert (a.shape, b.shape, c) == ((0, 0), (0,), 0)
    a, b, c = total_activation_matrix(np.empty((3, 3)), [], False)
    assert (a.shape, b.shape, c) == ((0, 0), (0,), 4.5)
    a, b, c = total_activation_matrix(np.empty((2, 3)), [np.array([[0, 1]])], False)
    assert_array_equal(a, [[.5]])
    assert_array_equal(b, [-2])
    assert c == 2
    a, b, c = total_activation_matrix(np.empty(6), [np.empty(3)], False)
    assert_array_equal(a, [[.5, .5, .5],
                           [.5, .5, .5],
                           [.5, .5, .5]])
    assert_array_equal(b, [-6, -6, -6])
    assert c == 18



def test_total_activation_energy():
    a, b, c = total_activation_matrix_(0, 0, False)
    assert total_activation_energy(a, b, c, np.array([])) == 0
    a, b, c = total_activation_matrix_(3, 0, False)
    assert total_activation_energy(a, b, c, np.array([])) == 4.5
    a, b, c = total_activation_matrix_(6, 1, False)
    assert total_activation_energy(a, b, c, np.array([0])) == 18
    assert total_activation_energy(a, b, c, np.array([1])) == 12.5
    assert total_activation_energy(a, b, c, np.array([3])) == 4.5
    assert total_activation_energy(a, b, c, np.array([6])) == 0
    assert total_activation_energy(a, b, c, np.array([7])) == 0.5
    assert total_activation_energy(a, b, c, np.array([9])) == 4.5
    a, b, c = total_activation_matrix_(4, 4, False)
    assert total_activation_energy(a, b, c, np.array([0, 0, 0, 0])) == 8
    assert total_activation_energy(a, b, c, np.array([1, 1, 1, 1])) == 0
    assert total_activation_energy(a, b, c, np.array([1, 1, 0, 0])) == 2
    assert total_activation_energy(a, b, c, np.array([.5, .5, .5, .5])) == 2


def test_count_vertices_gradient():
    a, b, c = total_activation_matrix_(0, 0, False)
    assert_array_equal(total_activation_energy_gradient(a, b, np.array([])), np.array([]))
    a, b, c = total_activation_matrix_(3, 0, False)
    assert_array_equal(total_activation_energy_gradient(a, b, np.array([])), np.array([]))
    # assert total_activation_energy_gradient(a, b, np.array([0])) == -6
    # assert total_activation_energy_gradient(6, 2.) == -4
    # assert total_activation_energy_gradient(6, 6.) == 0
    # assert total_activation_energy_gradient(6, 8.) == 2
