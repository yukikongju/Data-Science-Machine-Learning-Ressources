#  We generate random index and swap with adjacent

import random


class RandomWordSwap(BaseEstimator, TransformerMixin):

    def __init__(self, num_duplicates, num_swaps):
        """ 
        Arguments
        ---------
        num_duplicates: int
            how many time we duplicate dataset
        num_swaps: int
            num of token to swap
        """
        self.num_duplicates = num_duplicates
        self.num_swaps = num_swaps

    def fit(self, X, y):
        return self

    def transform(self, documents, labels):
        # 1. duplicate documents
        documents, labels = list(documents), list(labels)
        duplicate_docs = [ sentence for i in range(self.num_duplicates) 
                for sentence in documents]
        duplicate_labels = [ label for i in range(self.num_duplicates) 
                for label in labels]

        new_docs = []
        # 2. perform word swap
        for sentence, label in zip(duplicate_docs, duplicate_labels):
            new_sentence = self._swap_words_in_sentence(sentence)
            new_docs.append([new_sentence, label])

        return new_docs

    def _swap_words_in_sentence(self, sentence):
        """ 
        Given a sentence (string), return a new sentence where words were 
        swap n times
        """
        words = sentence.split(' ')
        for _ in range(self.num_swaps):
            # generate random index swap
            index = random.randrange(len(words) - 1)
            words[index], words[index + 1] = words[index + 1], words[index]
        return " ".join(words)


class RandomCharacterSwap(BaseEstimator, TransformerMixin):

    def __init__(self, num_duplicates, num_swaps):
        """ 
        Arguments
        ---------
        num_duplicates: int
            how many time we duplicate dataset
        num_swaps: int
            num of token to swap
        """
        self.num_duplicates = num_duplicates
        self.num_swaps = num_swaps

    def fit(self, X, y):
        return self

    def transform(self, documents, labels):
        pass

