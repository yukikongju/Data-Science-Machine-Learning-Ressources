 #  We are trying to build a topic vector from tf-idf. For this example, 
 #  the weight and the topics will be manually chosen
 #  sentence = {'topic1': 0.3, 'topic2': 0.4, ...}
 #  topic['petness'] = (0.3 * tfidf['cat'] + 0.3 * tfidf['dog'] - 0.5 * tfidf['apple'])
 #  We can use matrix multiplication to get our vector topic

#  Steps:
#  1. Get co-occurence matrix
#  2. transform co-occurence matrix to TF-IDF
#  3. Define Topic Matrix
#  4. Get Topic Vector with matrix mult: TF-IDF mat x Topic Matrix

import nltk
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin



class TFIDFVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.sentences = None
        self.vocabs = None
        self.word_dict = None

    def _get_word_dict(self, vocabs):
        word_dict = {word: i for i, word in enumerate(vocabs)}
        return word_dict
        

    def _get_vocab(self, sentences):
        vocabs = list(set([ word.lower() for sentence in sentences for word in sentence.split(' ') ]))
        vocabs.sort()
        return vocabs


    def get_occurence_mat(self, sentences, vocabs, word_dict):
        """ 
        Get co-occurence matrix from list of sentences

        Returns
        -------
        df: pandas dataframe
            
        """

        # count frequency for each sentence
        occ_mat = []
        for sentence in sentences: 
            frequency = [0 for _ in range(len(vocabs))]
            for word in sentence.split(' '):
                frequency[word_dict.get(word.lower())] += 1

            occ_mat.append(frequency)

        # create dataframe
        df = pd.DataFrame(occ_mat, columns=vocabs)

        return df

    def get_one_hot_encoding(self, sentences, vocabs, word_dict):
        """ 
        Get one hot encoding. Size = len(vocab) x len(vocab)
        """
        # count frequency for each sentence
        one_hot_mat = []
        for sentence in sentences: 
            frequency = [0 for _ in range(len(vocabs))]
            for word in sentence.split(' '):
                frequency[word_dict.get(word.lower())] = 1

            one_hot_mat.append(frequency)

        # create dataframe
        df = pd.DataFrame(one_hot_mat, columns=vocabs)

        return df

    def tf(self, sentences, vocabs, word_dict):
        """ 
        Get term frequency mat. Equivalent to getting co-occurence mat as numpy
        """
        return self.get_occurence_mat(sentences, vocabs, word_dict).to_numpy()
        

    def idf(self, sentences, vocabs, word_dict):
        """ 
        Compute IDF from co-occurence matrix: IDF size = 1 x vocab

        Params
        ------
        mat: array
            co-occurence matrix

        Attributes
        ----------
        doc_freq: array of size 1 x vocab len
        """
        one_hot_mat = self.get_one_hot_encoding(sentences, vocabs, word_dict)
        doc_freq = one_hot_mat.sum(axis = 0) # sum of each columns
        idf = round(doc_freq / len(sentences), 2)
        idf_array = np.array(idf).T
        return idf_array

    def tfidf(self, sentences):
        """ 
        TF-IDF size: num sentences x vocab length
            
        Parameters
        ----------
        sentences: list of string

        Returns
        -------
        """
        vocabs = self._get_vocab(sentences)
        word_dict = self._get_word_dict(vocabs)

        return self.tf(sentences, vocabs, word_dict) * self.idf(sentences, vocabs, word_dict)

    def fit_transform(self, X):
        return self.tfidf(X)
        
class TopicVectorizer(BaseEstimator, TransformerMixin):

    """ 
    Topic Vectorizer = TF-IDF x Topic Matrix

    TF-IDF size: num sentences x vocab length
    Topic Matrix size: vocab length x num topic

    """

    def __init__(self):
        pass

        
if __name__ == "__main__":
    sentences = [
        "We usually go to the supermarket to get cat food", 
        "I went to the zoo this monday with my class", 
        "I was studying for my economics exam"
    ]

    topics = [
        "animal", 
        "education", 
        "hobbies", 
        "food"
    ]
    vectorizer = TFIDFVectorizer()
    print(vectorizer.fit_transform(sentences))

    # TODO: to get the topic vector: multiply TF-IDF mat by topic matrix of 
    # size len(vocab) x num topic

