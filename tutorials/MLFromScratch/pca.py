
# Question: do we find PCA iteratively or do we select the top 

# Idea: we want to find V, the transformation matrix we use to transform the 
#   data into a lower subspace. we construct V by taking eighten vectors sorted 
#   by their eighten value in decreasing order. The eigthen values and 
#   vectors are calculated using the covariance matrix


import numpy as np

from sklearn.datasets import load_iris
from scipy.linalg import eigh

class PCA():

    def __init__(self, n_components):
        """ 
        Parameters
        ----------
        n_components: 
            num of dimension after reduction
        """
        self.n_components = n_components

    def fit_transform(self, data):
        """ 
        Parameters 
        ----------
        data: numpy array
            dataset
        """
        # 0. init variables
        self.num_cols = len(data[0])


        # 1. center the data
        means = np.mean(data, axis=0)   # moyenne des colonnes
        X = data - means

        # 2. calculer la matrice de covariance
        cov_mat = self._get_covariance_matrix(X)

        # 3. calculate eighten values and vectors + sort by highest eigthen value
        eig_values, eig_vectors = eigh(cov_mat)     # solve det(A-lambdaI) = 0
        idx_order = np.argsort(eig_values)[::-1]
        eig_values = eig_values[idx_order]
        eig_vectors = eig_vectors[idx_order]
        V = np.transpose(eig_vectors)

        # 4. transformed data: PCs => X' = XV
        PCs = np.matmul(X, V)

        # 5. calculate percentage of variance captured by each PC
        pc_variances = np.var(PCs, axis=0)
        perc_pc_variances = pc_variances / np.sum(pc_variances)

        # 6. FIXME: drop unrequired components? (or should be iterative?)
        X_transformed = PCs[:, :self.n_components]
        
        return X_transformed

    def _get_covariance_matrix(self, X):
        #  cov_mat = np.cov(np.transpose(X))
        num_cols = len(X[0])
        n = len(X)
        cov_mat = np.zeros((num_cols, num_cols))

        # note: covariance matrix is symetric
        for i in range(num_cols):
            for j in range(i, num_cols):
                #  cov = sum( (xi-mean(x)) (yi-mean(yi))) / N-1
                stx = X[:, i] - np.mean(X[:, i], axis=0)
                sty = X[:, j] - np.mean(X[:, j], axis=0)
                cov = np.sum(stx*sty) / (n-1)
                cov_mat[i][j] = cov_mat[j][i] = cov

        return cov_mat


#  ------------------------------------------------------------------------

def test_pca_2d():
    data = np.array([[126, 78], [128, 80], [128, 82], [130, 82], [130, 84], [132, 86]])
    pca = PCA(2)
    print(pca.fit_transform(data))
    
def test_pca_iris():
    data = load_iris().data
    pca = PCA(3)
    print(pca.fit_transform(data))

#  ------------------------------------------------------------------------


def main():
    #  test_pca_2d()
    test_pca_iris()
    


if __name__ == "__main__":
    main()
