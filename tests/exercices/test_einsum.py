import numpy as np
import pytest

from exercices.tensor_operations.einsum import dot_product, vector_sum, matrix_vector_product, matrix_matrix_multiplication

@pytest.fixture
def a():
    return np.random.randint(1, 10, (3))

@pytest.fixture
def b():
    return np.random.randint(1, 10, (3))

@pytest.fixture
def A():
    return np.random.randint(1, 10, (3, 3))

@pytest.fixture
def B():
    return np.random.randint(1, 10, (3, 5))

def test_vector_sum(a):
    x1 = a.sum()
    x2 = vector_sum(a)
    assert x1 == x2

def test_dot_product(a, b):
    x1 = np.dot(a, b)
    x2 = dot_product(a, b)
    assert x1 == x2

def test_matrix_vector_product(A, b):
    x1 = np.matmul(A, b)
    x2 = matrix_vector_product(A, b)
    x3 = A @ b
    assert x1.shape == x2.shape == x3.shape == (A.shape[0], )
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(x1, x3)

def test_matrix_matrix_multiplication(A, B):
    x1 = np.matmul(A, B)
    x2 = matrix_matrix_multiplication(A, B)
    x3 = A @ B
    assert x1.shape == x2.shape == x3.shape == (A.shape[0], B.shape[1])


