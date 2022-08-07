import sys 

from gensim.models import Word2Vec, FastText, Doc2Vec
from gensim.test.utils import common_texts # some example sentences

new_sentences = [
    ['computer', 'aided', 'design'],
    ['computer', 'science'],
    ['computational', 'complexity'],
    ['military', 'supercomputer'],
    ['central', 'processing', 'unit'],
    ['onboard', 'car', 'computer'],
]


#  self.model.build_vocab(new_sentences, update=True) 

class Word2VecVectorizer(object):

    def __init__(self):
        self.model = Word2Vec(sentences=common_texts, vector_size=25, window=5, min_count=1, epochs=10)

    def update_vocab(self, new_sentences): # get word vect for word not trained with
        self.model.build_vocab(new_sentences, update=True) 
        self.model.train(new_sentences, total_examples=len(new_sentences), epochs=self.model.epochs)

        
    def get_word_vect(self, word):
        return self.model.wv[word]

    def is_word_in_vectorizer(self, word):
        return word in self.model.wv.key_to_index

    def get_word_similarity(self, word1, word2):
        return self.model.wv.similarity(word1, word2)
        


def test_word_in_corpus(): # test function with words that have been trained on
    print(common_texts)
    word1, word2 = 'computer', 'response'
    word2vec = Word2VecVectorizer()
    print(word2vec.is_word_in_vectorizer(word1))
    print(word2vec.get_word_vect(word1))
    print(word2vec.get_word_similarity(word1, word2))


if __name__ == "__main__":
    test_word_in_corpus()
