import numpy as np
import pytest

from exercices.tensor_operations.einsum import dot_product, vector_sum

@pytest.fixture
def a():
    return np.random.randint(1, 10, (10))

@pytest.fixture
def b():
    return np.random.randint(1, 10, (10))

@pytest.fixture
def A():
    return np.random.randint(1, 10, (2, 3))

@pytest.fixture
def B():
    return np.random.randint(1, 10, (2, 3))

def test_vector_sum(a):
    x1 = a.sum()
    x2 = vector_sum(a)
    assert x1 == x2


def test_dot_product(a, b):
    x1 = np.dot(a, b)
    x2 = dot_product(a, b)
    assert x1 == x2




