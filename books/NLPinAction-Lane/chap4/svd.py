#  How does SVD decomposition works 
#       Formula: W_{m x n} = U_{m x p} S_{p x p} V_{p x n} ^T
#       1. Find co-occuring words (ie words that belong together) by calculating
#           correlation between tokens. Group highly correlated token together
#       
#   What does each matrix decomp means
#   - U mat: cross-correlation matrix between words and topics
#   - S mat: singular value => how much information is captured by each dimension
#               (it is a diagonal matrix)
#   - V mat: shared meaning between documents. measure how often documents 
#               use the same topics in new semantic model

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Step 1: create dummy documents

dummy_file = "books/NLPinAction-Lane/chap4/dummy_text.txt"
with open(dummy_file, 'r') as f:
    documents = f.read()

documents = documents.split('.\n')
print(documents)


# Step 2: get co-occurence matrix with CountVectorizer

vectorizer = CountVectorizer()
documents_vectorized = vectorizer.fit_transform(documents)
occ_mat = documents_vectorized.toarray()
print(occ_mat)
print(occ_mat.shape)
word_dict = vectorizer.vocabulary_
print(word_dict)

# Step 3: SVD Decomp

U, S, Vt = np.linalg.svd(occ_mat)
print(U.round(2))                   # 11 x 11
print(S.round(2))                   # 1  x 11
print(Vt.round(2))                  # 81 x 81

# TODO: Step 4: Truncating the topics


