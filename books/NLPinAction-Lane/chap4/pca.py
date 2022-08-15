
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE


# Step 1: get dummy documents

dummy_file = "books/NLPinAction-Lane/chap4/dummy_text.txt"
with open(dummy_file, 'r') as f:
    documents = f.read()
documents = documents.split('.\n')[:-1] # remove last sentences because empty
print(documents)

# Step 2: get TF-IDF co-occurence matrix

#  vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()
documents_vectorized = vectorizer.fit_transform(documents)
occ_mat = documents_vectorized.toarray()
word_dict = vectorizer.vocabulary_
vocab = sorted(vectorizer.vocabulary_.keys())

# Step 3: Topic vectorization with PCA => size: num sentences x num topics

n_topics = 6
pca = PCA(n_components=n_topics)
pca_topic_vectors = pca.fit_transform(occ_mat)
print(pca_topic_vectors)

# Step 4: Get most interesting topics

weights = pd.DataFrame(pca.components_.round(4), columns=word_dict, 
        index=[f'topic{i}' for i in range(n_topics)])
print(weights.T)
#  print(weights.T.sum())

# Step 5: Visualize topic vect
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

plot_embedding(weights.T)
    


