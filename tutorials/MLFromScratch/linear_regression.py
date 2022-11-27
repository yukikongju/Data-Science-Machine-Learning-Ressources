# formula: Y = XB
# calculating the parameters:    B = (X'X)^-1 * X' * y; B = [b a1 a2 ...]
# link: https://www.stat.purdue.edu/~boli/stat512/lectures/topic3.pdf
# Idea: when training the linear regression, we are trying to find the weights 
#   (slope) and the bias which minimizes the sum of squared residuals. 

# gradient descent: https://github.com/matheusgomes28/gradient-descent-notebook/blob/main/GradientDescent.ipynb


# We will be comparing the results of linear regression 3 ways: 
# (1) using Matrix parameters estimation 
# (2) using iterative method to minimizes SSR 
# (3) sklearn library


import numpy as np
import pandas as pd

from scipy.linalg import inv

from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression


def mse(y_true, y_pred):
    return np.mean(np.sum((y_true - y_pred)**2))
    

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
    

class LinearRegressionIterative(): 

    def __init__(self, lr=1e-3, num_iters=100000):
        self.lr = lr
        self.num_iters = num_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # initialization
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # update the weights iteratively
        for _ in range(self.num_iters):
            # 1. get predictions using current parameters
            y_pred = np.dot(X, self.weights) + self.bias

            # 2. calculate gradients
            d_weights = (-2/num_samples) * np.dot(X.T, (y - y_pred))
            d_bias = (-2/num_samples) * np.sum(y - y_pred)

            # 3. update the parameters
            self.weights = self.weights - self.lr * d_weights
            self.bias = self.bias - self.lr * d_bias

        
    def predict(self, X):
        # predictions
        predictions = np.dot(X, self.weights) + self.bias

        return predictions


#  ------------------------------------------------------------------------

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

    
def test_gradient_descent_linear_reg(features, targets):
    model = LinearRegressionIterative()
    model.fit(features, targets)
    predictions = model.predict(features)
    return predictions

#  ------------------------------------------------------------------------


def main():
    #  features, targets = load_boston()[:15]
    features, targets = make_regression(n_samples=10, n_features=3)

    # get predictions with each methods
    sklearn_predictions = test_sklearn_linear_reg(features, targets)
    matrix_predictions = test_matrix_linear_reg_custom(features, targets)
    gradient_predictions = test_gradient_descent_linear_reg(features, targets)

    # compare results
    print(sklearn_predictions)
    print(matrix_predictions)
    print(gradient_predictions)

    # compare errors
    print(f"Erreurs sklearn: {mse(targets, sklearn_predictions)}")
    print(f"Erreurs matrix: {mse(targets, matrix_predictions)}")
    print(f"Erreurs gradient: {mse(targets, gradient_predictions)}")
    
    

if __name__ == "__main__":
    main()

