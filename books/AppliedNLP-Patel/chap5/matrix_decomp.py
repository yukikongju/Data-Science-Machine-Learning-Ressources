import numpy as np 
import pandas as pd

from scipy.linalg import lu, lu_factor, lu_solve
from scipy.linalg import cholesky, qr


def lu_decomp(A):
    """ 
    A = LU
    Remark: we can get cholesky decomp for all squared matrix
    """
    pass
    


def test_lu_decomp():
    #  A = np.random.randint(0, 5, (6,6))
    A = np.array([ [4,-2,-3,1], [1,3,1,3], [1,2,-1,-2], [2,1,-1,-1] ])
    P, L, U = lu(A)
    print(A)
    print(L @ U)
    print(P@L@U)
    
def test_choleskey_decomp(A):
    U = cholesky(A)
    print(A)
    print(U.T @ U)
    
def test_qr_decomp():
    #  A = np.random.randint(0, 5, (6,6))
    A = np.array([ [4,-2,-3,1], [1,3,1,3], [1,2,-1,-2], [2,1,-1,-1] ])
    Q, R = qr(A)
    print(Q)
    print(Q @ R)
    print(A)

    

def get_co_occurence_matrix(sentences):
    """ 
    Get co-occurence matrix

    Param
    -----
    sentences: list of string
        ie [['i', 'ate', 'sushi'], ['we', 'went', 'to', 'boston']]

    Returns
    -------
    df: pandas dataframe object
        co occurence matrix
    """
    # get all tokens
    tokens = list(set([word for sentence in sentences for word in sentence]))
    tokens.sort()

    token_idx = {token: i for i, token in enumerate(tokens)}

    # count words frequency by sentence
    tokens_freq = []
    for sentence in sentences:
        freq = [ 0 for _ in tokens ]
        for word in sentence: 
            freq[token_idx.get(word)] += 1
        tokens_freq.append(freq)

    # get co occurence matrix
    df = pd.DataFrame(tokens_freq, columns=tokens)

    return df



def main():
    pass

if __name__ == "__main__":
    #  test_lu_decomp()
    #  test_choleskey_decomp()
    #  test_qr_decomp()

    sentences = [['i', 'ate', 'sushi'], ['we', 'went', 'to', 'boston']]
    co_occurence_mat = get_co_occurence_matrix(sentences)
    print(co_occurence_mat)

