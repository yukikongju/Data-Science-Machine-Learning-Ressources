
# We will compare 3 methods:
# (1) sklearn
# (2)
# (3) gradient descent

import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


def get_classification_score(y_true, y_pred): 
    conf_mat = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_mat.ravel()
    print(conf_mat)

    accuracy = (tn + tp) / (tn + fn + tp + fp)
    sensitivity = tp / (tp + fn)        # true positive: tp / t
    specificity = tn / (tn + fp)        # true negative: tn / n
    precision = tp / (tp + fp)          # tp / p

    scores = {'accuracy': accuracy, 'sensitivity': sensitivity, 
              'specificity': specificity, 'precision': precision }

    return scores


class LogisticRegressionGradientDescent(): # TODO:

    def __init__(self, lr=1e-4, num_iters=6000):
        self.lr = lr
        self.num_iters = num_iters

    def fit(self, ):
        pass
        
    def predict(self, ):
        pass
        


#  ------------------------------------------------------------------------



def test_sklearn_log_reg(features, targets):
    model = LogisticRegression()
    model.fit(features, targets)
    predictions = model.predict(features)
    return predictions
    

#  ------------------------------------------------------------------------

def main():
    features, targets = make_classification(n_samples=100, n_features=4, n_classes=2)

    # get predictions for each methods
    sklearn_pred = test_sklearn_log_reg(features, targets)


    # compare predictions score
    print(get_classification_score(targets, sklearn_pred))
    


if __name__ == "__main__":
    main()


