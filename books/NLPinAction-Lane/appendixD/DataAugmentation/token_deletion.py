# We want to duplicate our dataset n times and perform m random deletion

import numpy as np
import pandas as pd
import random


class RandomWordDeletion(BaseEstimator, TransformerMixin):

    def __init__(self, num_duplicates, num_deletions):
        """ 
        Arguments
        ---------
        num_duplicates: int
            how many time we duplicate dataset
        num_deletions: int
            num of token to delete
        """
        self.num_duplicates = num_duplicates
        self.num_deletions = num_deletions

    def fit(self, X, y):
        return self

    def transform(self, documents, labels):
        # 1. duplicate documents
        documents, labels = list(documents), list(labels)
        duplicate_docs = [ sentence for i in range(self.num_duplicates) 
                for sentence in documents]
        duplicate_labels = [ label for i in range(self.num_duplicates) 
                for label in labels]

        # 2. perform word deletion
        new_docs = []
        for sentence, label in zip(duplicate_docs, duplicate_labels):
            words = sentence.split(' ')
            num_words_removed = 0
            while words and num_words_removed < self.num_deletions:
                # remove random index
                index_to_remove = random.randrange(len(words))
                words.pop(index_to_remove)
                num_words_removed += 1
            new_sentence = " ".join(words)
            if new_sentence: # make sure that sentence is at least one word 
                new_docs.append([new_sentence, label])

        return new_docs
        
        
class RandomCharacterDeletion(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        pass

