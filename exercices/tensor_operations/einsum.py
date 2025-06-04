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

def batch_matrix_trace(A):
    """
    Input: 
    - A: (B, N, N)


    """
    return np.einsum('bii,bii->b', A)


def batch_dot_product(a, b):
    """
    Input:
    - a: (B, N)
    - b: (B, N)

    Output:
    - c: (B, )
    """
    return np.einsum('bi,bi->b', a, b)

def batch_outer_product(a, b):
    """
    Input:
    - a: (B, M)
    - b: (B, N)

    Output:
    - c: (B, M, N)
    """
    return np.einsum('bi,bj->bij', a, b)


def bilinear_form(x, A):
    """
    Input:
    - x: (N, )
    - A: (N, N)

    Output:
    - s: scalar product sum 

    Definition: x^t * A * x
    """
    pass


def attention_mechanism(Q, K):
    """
    # Q: (B, H, T_q, D)
    # K: (B, H, T_k, D)
    np.einsum('bhtd,bhsd->bhts', Q, K)
    """
    pass

def pairwise_distance(X, Y):
    """
    #  X: (N, D)
    #  Y: (M, D)
    #  Goal: squared Euclidean distance: ||x - y||² = x² + y² - 2xy
    xx = np.einsum('nd,nd->n', X, X)  # (N,)
    yy = np.einsum('md,md->m', Y, Y)  # (M,)
    xy = np.einsum('nd,md->nm', X, Y)  # (N, M)
    dists = xx[:, None] + yy[None, :] - 2 * xy  # shape (N, M)
    """
    pass

def tensor_contraction(T, M):
    """
# T: (A, B, C)
# M: (C, D)
np.einsum('abc,cd->abd', T, M)
    """
    pass

def gram_matrix(X):
    """
# X: (N, D) — N data points, D features
np.einsum('nd,ne->de', X, X)
    """
    pass

def batch_jacobian_vector_product(J, v):
    """
Per-sample Jacobian-Vector Product
# J: (B, O, I) — batch of Jacobians
# v: (B, I) — vector for each sample
np.einsum('boi,bi->bo', J, v)
    """
    pass

def color_transform_images(img, T):
    """
# img: (H, W, 3) — RGB image
# T: (3, 3) — color transform matrix
np.einsum('hwc,cd->hwd', img, T)
    """
    pass

def image_correlation(patches1, patches2):
    """
# patches1, patches2: (N, K, K) — N patches of size K×K
np.einsum('nkl,nkl->n', patches1, patches2)
    """
    pass

def cross_correlation(a, b):
    """

# a, b: (B, T)
np.einsum('bt,bt->b', a, b)
    """
    pass

def weighted_sum_with_broadcasting(a, w):
    """
# a: (B, T, D) — values
# w: (B, T) — weights
np.einsum('btd,bt->bd', a, w)
    """
    pass

