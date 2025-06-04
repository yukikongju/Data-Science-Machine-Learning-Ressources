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
    Words: For each row 'i' in matrix A, multiply b[j]
    """
    return np.einsum('ij,j->i', A, b)

def matrix_matrix_multiplication(A, B):
    """
    Inputs:
    - A: matrix of shape (M, N)
    - B: matrix of shape (N, P)

    Outputs:
    - C: matrix of shape (M, P)

    Definition: C_ij = Ai0 * B0j + Ai1 * B1j + ... + Ain * Bnj
    """
    return np.einsum('ik,kj->ij', A, B)

def outer_product_vectors(a, b):
    """
    Inputs:
    - a: vector of shape (M, )
    - b: vector of shape (N, )

    Outputs:
    - C: Matrix of shape (M, N)

    Definition: C_ij = a_i * b_j
    """
    return np.einsum('i,j->ij', a, b)


def element_wise_matrix_multiplication(A, B):
    """
    Equivalent to A * B

    """
    return np.einsum('ij,ij->ij', A, B)

def matrix_trace(A):
    return np.einsum('ii->', A)

def matrix_transpose(A):
    return np.einsum('ij->ji', A)

def sum_over_matrix_row(A):
    """
    Input:
    - A: matrix of shape (M, N)

    Output:
    - C: vector of shape (M, )

    Definition: C_i = Ai0 + Ai1 + Ai2 + .. + Ain
    """
    return np.einsum('ij->i', A)

def sum_over_matrix_column(A):
    """
    Input:
    - A: matrix of shape (M, N)

    Output:
    - C: vector of shape (N, )

    Definition: C_j = A0j + A1j + A2j + .. + Anj
    """
    return np.einsum('ij->j', A)

def batch_matrix_multiplication(A, B):
    """
    Input: 
    - A: matrix of shape (B, M, K)
    - B: matrix of shape (B, K, N)

    Output:
    - C: matrix of shape (B, M, N)

    Definition: Given bth batch, compute A[b] * B[b]
    """
    return np.einsum('bik,bkj->bij', A, B)

