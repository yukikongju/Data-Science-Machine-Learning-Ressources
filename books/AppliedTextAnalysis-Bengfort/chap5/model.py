import pandas as pd
import numpy as np

from collections import defaultdict

from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from nltk.stem import SnowballStemmer

from loader import CSVLoader

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier


def create_pipeline(model, reduction=False):
    steps = []

    if reduction:
        steps.append((
            'reduction', TruncatedSVD(n_components=1000)
            ))

    steps.append(('classifier', model))
    return Pipeline(steps)
    


def main():
    # load data
    stemmer = SnowballStemmer('english')
    #  file_path = 'data/'
    #  file_path = 'books/AppliedTextAnalysis-Bengfort/chap5/data/SMS_train.csv'
    loader = CSVLoader(file_path, stemmer, 'Message_body', 'Label')
    X, y = loader.get_data()

    # pipeline
    models = []
    for form in [LogisticRegression, SGDClassifier, KNeighborsClassifier]:
        models.append(create_pipeline(form(), True))
        models.append(create_pipeline(form(), False))

    # train test data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # train and test models
    scores = defaultdict(list)

    for model in models:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        #  score = accuracy_score(y_test, y_pred)
        #  scores.append(score)
        scores['model'].append(str(model))
        scores['accuracy'].append(accuracy_score(y_test, y_pred))

        scores['precision'].append(precision_score(y_test, y_pred, pos_label='Spam'))
        scores['recall'].append(recall_score(y_test, y_pred, pos_label='Spam'))
        scores['f1'].append(f1_score(y_test, y_pred, pos_label='Spam'))

        print(f"Classification Report for {model}")
        print(classification_report(y_test, y_pred))

    print(dict(scores))



if __name__ == "__main__":
    main()
