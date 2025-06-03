import numpy as np

"""
Rules:
    1) Repeating letters in different inputs means those values will be 
       multiplied and those products will be the outputs
       Ex: np.einsum('ik,kj->ij', A, B)
    2) Omitting letters means that axis will be summed
       Ex: np.einsum('i->', x)
    3) We can return the unsummed axis in any order
       Ex: np.einsum('ijk->kji', x)
"""

def vector_sum(a):
    """
    a is a vector of shape (n, )

    Definition: a1 + a2 + ... + an
    """
    return np.einsum('i->', a)

def dot_product(a, b):
    """
    a, b are vectors of shape (n, )

    Definition: a1*b1 + a2*b2 + ... + an * bn
    """
    return np.einsum('i,i->', a, b)

def matrix_vector_product(A, b):
    """
    Inputs:
    - A: matrix of shape (M, N)
    - b: vector of shape (N, )

    Output:
    - C: vector of shape (M, )

    Definition: A1 * b + A2 * b + ... + An * b

    """
    return np.einsum('ij,j->i', A, b)


def matrix_matrix_product(A, B):
    pass

def outer_product_vectors(a, b):
    pass

def batch_matrix_multiplication(A, B):
    pass

def element_wise_matrix_multiplication(A, B):
    """
    Equivalent to A * B

    """
    pass

def matrix_trace(A):
    pass

def matrix_transpose(A):
    pass

def sum_over_matrix_row(A):
    pass

def sum_over_matrix_column(A):
    pass


