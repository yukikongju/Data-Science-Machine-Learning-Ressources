import nltk
import string
import numpy as np

from collections import defaultdict
from abc import ABC

from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


class TextVectorizer(ABC):

    def __init__(self, stemmer, corpus):
        self.stemmer = stemmer
        self.corpus = corpus
        self.normalized_corpus = self.normalize_corpus()
        self.bag_of_words = self.get_bag_of_words()

    def get_data(self):
        """ 
        Turn corpus into numpy array

        Examples:
        >>> TextVectorizer(stemmer, corpus).get_data()
        >>> [[0,1,0,1], [0,0,1,0]]
        """
        corpus_vectorized = self.vectorize_corpus()
        data = [[val for val in counter.values()] for counter in corpus_vectorized]

        return data

    def vectorize_corpus(self):
        """ 
        Get Vectorized corpus
        """
        vect = []
        for sentence in self.normalized_corpus:
            vect.append(self.vectorize(sentence))
        return vect

    def vectorize(self, sentence):
        """ 
        Vectorize sentence
        """
        pass
        

    def normalize_corpus(self):
        """ 
        Return normalized corpus without stopwords, punctuation

        Parameters:
            corpus: list of strings
                list of sentences

        Returns:
            corpus: list of list of string
                tokenize and stemmed each sentence in corpus without punctuation 
                and stop words

        Examples:
        >>> corpus = ["I ate pizza", "I drank water"]
        >>> normalize_corpus(corpus)
        >>> [["ate", "pizza"], ["drank", "water"]]
        """
        corpus = [word_tokenize(sentence) for sentence in self.corpus]
        stop_words = set(stopwords.words('english'))
        #  stemmer = SnowballStemmer('english')

        new_corpus = []
        for sentence in corpus:
            new_sentence = []
            for word in sentence:
                word = word.lower()
                if word not in stop_words and word not in string.punctuation:
                    new_sentence.append(self.stemmer.stem(word))
            new_corpus.append(new_sentence)

        return new_corpus

    def get_bag_of_words(self):
        """ 
        Get bag of words from corpus (remove stop words and punctuation, 
        normalize text with stemmer)

        Parameters:
            corpus: list
                list of sentences

        Returns:
            bag: list
                bag of words

        Examples:
        >>> corpus = ["I ate pizza", "I drank water"]
        >>> get_bag_of_words(corpus)
        >>> [["ate", "drank", "pizza", "water"]]
        """
        corpus = [word_tokenize(sentence) for sentence in self.corpus]
        stop_words = set(stopwords.words('english'))
        #  stemmer = SnowballStemmer('english')

        # get all different words
        bag = []
        for sentence in corpus:
            for word in sentence:
                word = word.lower()
                if word not in stop_words and word not in string.punctuation:
                    bag.append(self.stemmer.stem(word))

        # remove dupplicates and sort by alphabetical order
        bag = list(set(bag))
        bag.sort()
        return bag
        

class OneHotEncodingVectorizer(TextVectorizer):

    def __init__(self, stemmer, corpus):
        super().__init__(stemmer, corpus)

    def vectorize(self, sentence):
        """ 
        Vectorize normalized corpus using: one hot encoding (1 if token in bag, else 0)
        
        Parameters: 
            sentence: list of string
                sentence tokenize
            bag_of_words: list of string
                all tokenize words in entire document

        Returns:
            distribution

        Examples:
        >>> sentence = ['ate', 'potato', 'potato']
        >>> bag_of_words = ['ate', 'bat', 'drink', 'potato']
        >>> frequency_vectorization(sentence, bag_of_words)
        >>> {'ate': True, 'bat': False, 'drink': False, 'potato': True}
        """
        #  initialize counter dict
        counter = {}
        for word in self.bag_of_words:
            counter[word] = 0 

        # frequency count
        for word in sentence:
            counter[word] = 1

        return counter


class FrequencyVectorizer(TextVectorizer):


    def __init__(self, stemmer, corpus):
        super().__init__(stemmer, corpus)

    def vectorize(self, sentence):
        """ 
        Vectorize normalized corpus using: frequency vectorization
        
        Parameters: 
            sentence: list of string
                sentence tokenize
            bag_of_words: list of string
                all tokenize words in entire document

        Returns:
            distribution

        Examples:
        >>> sentence = ['ate', 'potato', 'potato']
        >>> bag_of_words = ['ate', 'bat', 'drink', 'potato']
        >>> frequency_vectorization(sentence)
        >>> {'ate': 1, 'bat': 0, 'drink': 0, 'potato': 2}
        """
        #  initialize counter dict
        counter = {}
        for word in self.bag_of_words:
            counter[word] = 0

        # frequency count
        for word in sentence:
            counter[word] += 1

        return counter
        

def main():
    stemmer = SnowballStemmer('english')
    corpus = [
        "I ate a potato yesterday.", 
        "I went to the beach a month ago.",
        "I saw a bat drinking blood at the park."
    ]

    freq_vect = FrequencyVectorizer(stemmer, corpus)
    corpus_vectorized = freq_vect.vectorize_corpus() 
    data = freq_vect.get_data()
    print(data)

    #  onehot_vect = OneHotEncodingVectorizer(stemmer, corpus)
    #  corpus_vectorized = onehot_vect.vectorize_corpus()


if __name__ == "__main__":
    main()


