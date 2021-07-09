import numpy as np
from numpy.testing import assert_array_equal

from total import total_activation_matrix_, total_activation_energy, \
    total_activation_energy_gradient, total_activation_matrix


def test_total_activation_matrix_():
    a, b, c = total_activation_matrix_(0, 0)
    assert (a.shape, b.shape, c) == ((0, 0), (0,), 0)
    a, b, c = total_activation_matrix_(3, 0)
    assert (a.shape, b.shape, c) == ((0, 0), (0,), 9)
    a, b, c = total_activation_matrix_(2, 1)
    assert_array_equal(a, [[0]])
    assert_array_equal(b, [-4])
    assert c == 4
    a, b, c = total_activation_matrix_(6, 3)
    assert_array_equal(a, [[0, 1, 1],
                           [1, 0, 1],
                           [1, 1, 0]])
    assert_array_equal(b, [-12, -12, -12])
    assert c == 36

    a, b, c = total_activation_matrix_(0, 0, False)
    assert (a.shape, b.shape, c) == ((0, 0), (0,), 0)
    a, b, c = total_activation_matrix_(3, 0, False)
    assert (a.shape, b.shape, c) == ((0, 0), (0,), 9)
    a, b, c = total_activation_matrix_(2, 1, False)
    assert_array_equal(a, [[1]])
    assert_array_equal(b, [-4])
    assert c == 4
    a, b, c = total_activation_matrix_(6, 3, False)
    assert_array_equal(a, [[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]])
    assert_array_equal(b, [-12, -12, -12])
    assert c == 36


def test_total_activation_matrix():
    null_segment = np.empty((0, 2), dtype=int)
    a, b, c = total_activation_matrix(np.empty(0), null_segment)
    assert (a.shape, b.shape, c) == ((0, 0), (0,), 0)
    a, b, c = total_activation_matrix(np.empty((3, 3)), null_segment)
    assert (a.shape, b.shape, c) == ((0, 0), (0,), 9)
    a, b, c = total_activation_matrix(np.empty((2, 3)), np.array([[0, 1]]))
    assert_array_equal(a, [[0]])
    assert_array_equal(b, [-4])
    assert c == 4
    a, b, c = total_activation_matrix(np.empty(6), np.empty(3))
    assert_array_equal(a, [[0, 1, 1],
                           [1, 0, 1],
                           [1, 1, 0]])
    assert_array_equal(b, [-12, -12, -12])
    assert c == 36

    a, b, c = total_activation_matrix(np.empty(0), null_segment, False)
    assert (a.shape, b.shape, c) == ((0, 0), (0,), 0)
    a, b, c = total_activation_matrix(np.empty((3, 3)), null_segment, False)
    assert (a.shape, b.shape, c) == ((0, 0), (0,), 9)
    a, b, c = total_activation_matrix(np.empty((2, 3)), np.array([[0, 1]]), False)
    assert_array_equal(a, [[1]])
    assert_array_equal(b, [-4])
    assert c == 4
    a, b, c = total_activation_matrix(np.empty(6), np.empty(3), False)
    assert_array_equal(a, [[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]])
    assert_array_equal(b, [-12, -12, -12])
    assert c == 36


def test_total_activation_energy():
    a, b, c = total_activation_matrix_(0, 0, False)
    assert total_activation_energy(a, b, c, np.array([])) == 0
    a, b, c = total_activation_matrix_(3, 0, False)
    assert total_activation_energy(a, b, c, np.array([])) == 9
    a, b, c = total_activation_matrix_(6, 1, False)
    assert total_activation_energy(a, b, c, np.array([0])) == 36
    assert total_activation_energy(a, b, c, np.array([1])) == 25
    assert total_activation_energy(a, b, c, np.array([3])) == 9
    assert total_activation_energy(a, b, c, np.array([6])) == 0
    assert total_activation_energy(a, b, c, np.array([7])) == 1
    assert total_activation_energy(a, b, c, np.array([9])) == 9
    a, b, c = total_activation_matrix_(4, 4, False)
    assert total_activation_energy(a, b, c, np.array([0, 0, 0, 0])) == 16
    assert total_activation_energy(a, b, c, np.array([1, 1, 1, 1])) == 0
    assert total_activation_energy(a, b, c, np.array([1, 1, 0, 0])) == 4
    assert total_activation_energy(a, b, c, np.array([.5, .5, .5, .5])) == 4


def test_count_vertices_gradient():
    a, b, c = total_activation_matrix_(0, 0, False)
    assert_array_equal(total_activation_energy_gradient(a, b, np.array([])), np.array([]))
    a, b, c = total_activation_matrix_(3, 0, False)
    assert_array_equal(total_activation_energy_gradient(a, b, np.array([])), np.array([]))
    # assert total_activation_energy_gradient(a, b, np.array([0])) == -6
    # assert total_activation_energy_gradient(6, 2.) == -4
    # assert total_activation_energy_gradient(6, 6.) == 0
    # assert total_activation_energy_gradient(6, 8.) == 2
