
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE

class PCATopicVectorizer(BaseEstimator, TransformerMixin):

    """ 
    Transform co-occurence matrix / TF-IDF into topic vectors 

    Examples
    --------
    >>> documents = ["I am writing", "The word is short"]
    >>> topic_vectorizer = PCATopicVectorizer(TfidfVectorizer(), n_topics = 6)
    >>> df = topic_vectorizer.fit_transform(documents)
    >>>             topic0  topic1  topic2  topic3  topic4  topic5
    ...     rather  0.0044  0.0951 -0.1192  0.2258 -0.1503 -0.0067
    >>> df size: num words x num topics
    """

    def __init__(self, vectorizer, n_topics):
        self.vectorizer = vectorizer
        self.n_topics = n_topics
        self.pca = PCA(n_components=n_topics)

    def fit(self, X):
        return self
        
    def transform(self, documents):
        # get co-occurence matrix with vectorizer
        documents_vectorized = self.vectorizer.fit_transform(documents)
        occ_mat = documents_vectorized.toarray()
        word_dict = self.vectorizer.vocabulary_
        vocab = sorted(self.vectorizer.vocabulary_.keys())

        # get topic vetor with pca (dimension reduction)
        pca_topic_vectors = self.pca.fit_transform(occ_mat)
        weights = pd.DataFrame(self.pca.components_.round(4), columns=word_dict, 
                index=[f'topic{i}' for i in range(self.n_topics)])
        return weights.T
        
class TruncatedSVDTopicVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, vectorizer, n_topics):
        self.vectorizer = vectorizer
        self.n_topics = n_topics
        self.svd = TruncatedSVD(n_components=n_topics)

    def fit(self, X):
        return self


    def transform(self, documents):
        # get co-occurence matrix with vectorizer
        documents_vectorized = self.vectorizer.fit_transform(documents)
        occ_mat = documents_vectorized.toarray()
        word_dict = self.vectorizer.vocabulary_
        vocab = sorted(self.vectorizer.vocabulary_.keys())

        # get topic vetor with pca (dimension reduction)
        topic_vectors = self.svd.fit_transform(occ_mat)
        weights = pd.DataFrame(self.svd.components_.round(4), columns=word_dict, 
                index=[f'topic{i}' for i in range(self.n_topics)])
        return weights.T
        

def plot_embedding(df):
    """ 
    Plot word embedding => rows: words; col: word vector
    """
    labels = df.index
    tsne = TSNE(perplexity=40, n_components=2, init='pca')
    new_values = tsne.fit_transform(df.to_numpy())

    # get coordinates
    x, y = [], []
    for value in new_values: 
        x.append(value[0])
        y.append(value[1])

    # plot words 
    plt.figure(figsize=(20, 20))
    for i in range(len(df)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i], xy=(x[i], y[i]), xytext=(5,2), 
                textcoords='offset points', ha='right', va='bottom')
    plt.show()

def main():
    # Step 1: get data
    dummy_file = "books/NLPinAction-Lane/chap4/dummy_text.txt"
    with open(dummy_file, 'r') as f:
        documents = f.read()
    documents = documents.split('.\n')[:-1] # remove last sentences because empty

    # Step 2: visualize topic embedding for each method
    vectorizers = [
        ('PCA', PCATopicVectorizer(TfidfVectorizer(), n_topics=6)),
        ('TruncatedSVD', TruncatedSVDTopicVectorizer(TfidfVectorizer(), n_topics=6))
    ]
    for name, vectorizer in vectorizers:
        # get topic embedding 
        words_vectorized = vectorizer.fit_transform(documents)
        print(words_vectorized)

        # plot topic embedding
        plot_embedding(words_vectorized)
        

if __name__ == "__main__":
    main()
