import pandas as pd
import nltk
import numpy as np

from nltk.cluster import KMeansClusterer
from utils import TextNormalizer, OneHotVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.pipeline import Pipeline

class KMeansCluster(BaseEstimator, TransformerMixin):

    def __init__(self, k):
        self.k = k
        self.distance = nltk.cluster.euclidean_distance
        self.model = KMeansClusterer(self.k, self.distance, avoid_empty_clusters=True)

    def fit(self, sentences, labels=None):
        return self

    def transform(self, sentences):
        pass
        

class HierarchicalClustering(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, sentences, labels=None):
        pass

    def transform(self, sentences):
        pass
        

def test_kmeans_clusterer():
    pass
    

def test_agglomerative_clustering(sentences):
    pipeline = Pipeline([
        ('normalizer', TextNormalizer()), 
        ('vectorizer', OneHotVectorizer()),
    ])
    X = pipeline.fit_transform(sentences)
    model = AgglomerativeClustering()
    clusters = model.fit_predict(X)
    print(clusters)
    


def test_pipeline(sentences):
    cluster = KMeans(n_clusters=7, algorithm='auto')
    model = Pipeline([
        ('normalizer', TextNormalizer()), 
        ('vectorizer', OneHotVectorizer()),
        ('cluster', cluster)
    ])

    clusters = model.fit_transform(sentences)
    pred = model.predict(sentences)
    
if __name__ == "__main__":
    file_path = "data/headlines/headlines_subset.csv"
    df = pd.read_csv(file_path)
    sentences = df['headline_text'].tolist()


    #  test_pipeline(sentences)
    test_agglomerative_clustering(sentences)
    #  test_kmeans_clusterer()

