import string
import pandas as pd

from itertools import chain

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


class CSVLoader(object):


    def __init__(self, csv_file_path, stemmer, body_col, label_col):
        self.csv_file_path = csv_file_path
        self.stemmer = stemmer
        self.body_col = body_col
        self.label_col = label_col
        self.data = self._read_file()

    def _read_file(self):
        data = pd.read_csv(self.csv_file_path, encoding='latin1')
        return data

        
    def get_corpus(self):
        """ 
        Examples:
        >>> CSVLoader(csv_file_path, stemmer, body_col, label_col).get_corpus()
        >>> [['wat', 'time', 'u', 'finish'], ['just', 'glad', 'talk']]
        >>> 
        """
        sentences = self.data[self.body_col]
        sentences = list(map(word_tokenize, sentences))

        stop_words = set(stopwords.words('english'))

        corpus = []
        for sentence in sentences:
            tokens = []
            for word in sentence:
                tokenize_word = self._tokenize_word(word)
                if tokenize_word:
                    tokens.append(tokenize_word)
            corpus.append(tokens)
        
        return corpus

    def _tokenize_word(self, word):
        """ 
        Function that tokenize word with stemmer if it is not a punctuation, 
        a stop word or a number
        """
        stop_words = set(stopwords.words('english'))

        word = word.lower()
        if word not in string.punctuation and word not in stop_words and not word.isdigit():
            return self.stemmer.stem(word)
        else: 
            return ''


    def get_data(self):
        """ 
        Function that return vectorized features and labels as array
        
        Examples: 
        >>> features, labels = CSVLoader(csv_file_path, stemmer).get_data()
        >>> features = [[0,0,2,0], [0,0,0,4]]
        >>> labels = ['spam', 'non-spam']
        """

        # Step 1: get labels
        labels = self.data[self.label_col]

        # Step 2: get features (TODO)
        corpus = self.get_corpus()
        bag_of_words = self.get_bag_of_words()

        # count tokens frequency for each sentences in corpus
        sentences_vect = []
        for sentence in corpus:
            sentence_vect = self._sentence_vectorization(sentence, bag_of_words)
            sentences_vect.append(sentence_vect)

        # transform counter into array
        features = [[val for val in counter.values()] for counter in sentences_vect]

        return features, labels

    def _sentence_vectorization(self, sentence, bag_of_words):
        """ 
        Vectorize sentence
        """
        # init counter
        counter = {}
        for word in bag_of_words:
            counter[word] = 0

        # count occurence
        for word in sentence:
            counter[word] += 1

        return counter
        

    def get_bag_of_words(self):
        # get all words in corpus in a 1D list
        corpus = list(chain.from_iterable(self.get_corpus()))

        # remove all dupplicates and sort
        corpus = list(set(corpus))
        corpus.sort()

        return corpus
        
        

def main():
    stemmer = SnowballStemmer('english')
    file_path = 'books/AppliedTextAnalysis-Bengfort/chap5/data/SMS_train.csv'
    loader = CSVLoader(file_path, stemmer, 'Message_body', 'Label')
    body, labels = loader.get_data()
    

if __name__ == "__main__":
    main()
