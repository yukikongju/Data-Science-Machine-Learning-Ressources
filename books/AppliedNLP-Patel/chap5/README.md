# Chap 5 - Embedding: How Machines Understand Words

[Code](https://github.com/nlpbook/nlpbook/blob/main/ch05.ipynb)

- [X] Using `nn.Embedding` for words vectorization
- [X] Using `nn.Embedding` inside `nn.Model`
- [o] Training
    - [X] Using custom training function
    - [ ] Using `tqdm` to train model
- [ ] Testing
- [.] Explore pretrained embedding with `gensim.models`
    - [ ] Models: Word2Vec, FastText, Doc2Vec, GloVe
    - [X] With words trained with
    - [ ] With words that have not been trained with
- [ ] Build the vocabulary using `torchtext` TEXT
- [ ] Build Word Embedding using co-occurence matrix decomposition
    - [ ] Singular Value Decompostion, QR Decomposition, Choleskey, LU
    - [ ] Decomp as two matrices: 
	- [ ] Embedding is first/second matrices
	- [ ] Embedding is the log/mean of both matrices
- [ ] Build Embedding using TF-IDF rows


**Notes**
- What does `torch.nn.Embedding` does?
    * Create lookup dictionary
- FIXME: why embeddings works with `view()` but not with `reshape()`


## Ressources

**Word Embedding Classifiers**

- [ ] [Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)
- [ ] [torch.nn.Embedding Explained by midlyoverfitter](https://www.youtube.com/watch?v=euwN5DHfLEo)
- [ ] [Embedding Code Examples](https://www.programcreek.com/python/example/107677/torch.nn.Embedding)
- [X] [Word2Vec for text classification by Ethen181](https://ethen8181.github.io/machine-learning/keras/text_classification/word2vec_text_classification.html)

**Understanding what Embedding is**

- [X] [Pytorch Embeddings by Santerre AI](https://www.youtube.com/watch?v=mCvW_qNm7rY&t=763s)
- [X] [Embedding Progression by John Santerre](https://github.com/johnsanterre/Teaching/tree/main/embedding_progression)
- [X] [Intuition behind word embedding with co-occurence matrix decomposition by macheads101](https://www.youtube.com/watch?v=5PL0TmQhItY&t=578s)

**Gensim Models Documentation**


**Co-matrix decomposition**

- [ ] [Matrix Decompositions in Python](https://people.duke.edu/~ccc14/sta-663-2016/08_LinearAlgebraExamples.html)


