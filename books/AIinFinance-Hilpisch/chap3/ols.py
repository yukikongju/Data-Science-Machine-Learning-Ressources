#  We want to find OLS parameters (alpha and beta) using the following formula
#      beta = cov(x,y)/var(x)
#      alpha = mean(y) - beta * mean(x)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin


class OLSRegressor(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.alpha = None
        self.beta = None

    def fit(self, X, y):
        self.beta = self._get_beta(X, y)
        self.alpha = self._get_alpha(X, y, self.beta)

    def predict(self, X):
        return self.beta + self.alpha * X

    def _get_beta(self, X, y):
        return np.cov(X, y)[0,1] / np.var(X)

    def _get_alpha(self, X, y, beta):
        return np.mean(y) - beta * np.mean(X)
        
def plot_results(X, y_true, y_pred):
    plt.scatter(X, y_true)
    plt.scatter(X, y_pred)
    plt.show()
    

def main():
    # create dummy data
    n, low, high = 20, 1, 100
    x = np.random.uniform(low, high, (1,100))
    y = np.random.uniform(low, high, (1,100))
    ols = OLSRegressor()
    ols.fit(x, y)
    y_pred = ols.predict(x)

    # plot results
    plot_results(x, y, y_pred)


if __name__ == "__main__":
    main()

