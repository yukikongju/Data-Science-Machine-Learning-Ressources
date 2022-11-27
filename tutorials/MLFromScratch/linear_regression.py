# formula: Y = W^T * X + B
# calculating the parameters:    B = (X'X)^-1 * X' * y; B = [b a1 a2 ...]
# link: https://www.stat.purdue.edu/~boli/stat512/lectures/topic3.pdf
# Idea: when training the linear regression, we are trying to find the weights 
#   (slope) and the bias which minimizes the sum of squared residuals. 

# We will be comparing the results of linear regression 3 ways: 
# (1) using Matrix parameters estimation 
# (2) using iterative method to minimizes SSR 
# (3) sklearn library


import numpy as np
import pandas as pd

from scipy.linalg import inv
from sklearn.linear_model import LinearRegression

class LinearRegressionMatrix():

    def __init__(self):
        self.parameters = None

    def fit(self, X, y):
        n = len(X)
        self.num_variables = len(X[0])

        # 1. add col of 1 for bias: X = [1 x1 x2 ...]
        X = np.hstack((np.ones((n, 1)), X))

        # 2. calculating the parameters: B = (X^T*X)^-1 * X^T * y; B = [b a1 a2]
        self.parameters = inv(X.T @ X) @ X.T @ y

        
    def predict(self, X):
        n = len(X)
        # 1. add col of 1 for bias: X = [1 x1 x2 ...]
        X = np.hstack((np.ones((n, 1)), X))

        # 2. calculate predictions: y = a1 x1 + a2x2 + ... + b
        predictions = X @ self.parameters
        return predictions
    
    def _get_squared_errors(self): # e = y - f(x)
        pass

        

class LinearRegressionIterative():

    def __init__(self):
        pass

    def fit(self):
        pass
        
    def predict(self):
        pass


def load_boston():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    features = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    targets = np.array(raw_df.values[1::2, 2])
    return features, targets

#  ------------------------------------------------------------------------


def test_sklearn_linear_reg(features, targets):
    model = LinearRegression()
    model.fit(features, targets)
    predictions = model.predict(features)
    return predictions

def test_matrix_linear_reg_custom(features, targets):
    model = LinearRegressionMatrix()
    model.fit(features, targets)
    predictions = model.predict(features)
    return predictions

def test_matrix_linear_reg():
    Y = np.array([124, 95, 71, 45, 18])
    X = np.array([[49], [69], [89], [99], [109]])
    model = LinearRegressionMatrix()
    model.fit(X, Y)
    predictions = model.predict(X)
    print(predictions)
    

#  ------------------------------------------------------------------------




def main():
    features, targets = load_boston()[:15]

    # get predictions with each methods
    sklearn_predictions = test_sklearn_linear_reg(features, targets)
    print(sklearn_predictions)
    matrix_predictions = test_matrix_linear_reg_custom(features, targets)
    print(matrix_predictions)
    
    

if __name__ == "__main__":
    main()

