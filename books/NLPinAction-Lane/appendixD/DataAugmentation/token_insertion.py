# How to insert word: relative to word around or random word insertion?
# We will use wordnet to find related word
# later: use custom embedding

import pandas as pd 
import random


class RandomWordInsertion(BaseEstimator, TransformerMixin):

    def __init__(self, num_duplicates, num_insertions):
        """ 
        Arguments
        ---------
        num_duplicates: int
            how many time we duplicate dataset
        num_insertions: int
            num of token to insert
        """
        self.num_duplicates = num_duplicates
        self.num_deletions = num_insertions

    def fit(self, X, y):
        return self

    def transform(self, documents, labels):
        pass


class RandomCharacterInsertion(BaseEstimator, TransformerMixin):

    def __init__(self, num_duplicates, num_insertions):
        """ 
        Arguments
        ---------
        num_duplicates: int
            how many time we duplicate dataset
        num_insertions: int
            num of token to insert
        """
        self.num_duplicates = num_duplicates
        self.num_deletions = num_insertions

    def fit(self, X, y):
        return self

    def transform(self, documents, labels):
        pass

