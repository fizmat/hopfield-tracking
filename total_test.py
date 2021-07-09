import numpy as np
from numpy.testing import assert_array_equal

from reconstruct import energy, energy_gradient
from total import total_activation_matrix_, total_activation_matrix


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
    act = np.array([])
    assert energy(a, act) + b.dot(act) + c == 0
    a, b, c = total_activation_matrix_(3, 0, False)
    assert energy(a, act) + b.dot(act) + c == 9
    a, b, c = total_activation_matrix_(6, 1, False)
    act = np.array([0])
    assert energy(a, act) + b.dot(act) + c == 36
    act = np.array([1])
    assert energy(a, act) + b.dot(act) + c == 25
    act = np.array([3])
    assert energy(a, act) + b.dot(act) + c == 9
    act = np.array([6])
    assert energy(a, act) + b.dot(act) + c == 0
    act = np.array([7])
    assert energy(a, act) + b.dot(act) + c == 1
    act = np.array([9])
    assert energy(a, act) + b.dot(act) + c == 9
    a, b, c = total_activation_matrix_(4, 4, False)
    act = np.array([0, 0, 0, 0])
    assert energy(a, act) + b.dot(act) + c == 16
    act = np.array([1, 1, 1, 1])
    assert energy(a, act) + b.dot(act) + c == 0
    act = np.array([1, 1, 0, 0])
    assert energy(a, act) + b.dot(act) + c == 4
    act = np.array([.5, .5, .5, .5])
    assert energy(a, act) + b.dot(act) + c == 4


def test_count_vertices_gradient():
    a, b, c = total_activation_matrix_(0, 0, False)
    assert_array_equal(energy_gradient(a, np.array([])) + b, np.array([]))
    a, b, c = total_activation_matrix_(3, 0, False)
    assert_array_equal(energy_gradient(a, np.array([])) + b, np.array([]))
    a, b, c = total_activation_matrix_(6, 1, False)
    act = np.array([3])
    assert energy_gradient(a, act) + b == [2 * 1 * 3 - 12]
