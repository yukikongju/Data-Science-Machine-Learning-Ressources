#  We want to perform synonym replacement with words that are meaningful. To 
#  determine if a word if meaningful, we will use Zipft law


import pandas as pd

from itertools import chain

from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import wordnet
from textattack.augmentation import WordNetAugmenter, EmbeddingAugmenter


class CustomSynonymReplacement(BaseEstimator, TransformerMixin):

    """ 
    We want to perform synonym replacement with words that are meaningful. To 
    determine if a word if meaningful, we will use Zipft law
    """

    def __init__(self, max_synonyms = 3, perc_replacements = 0.025):
        """ 
        Parameters 
        ----------
        max_synonyms: int
            maximum of synonyms for each words
        perc_replacements: float
            percentage of words to be replaced
        """
        self.max_synonyms = max_synonyms
        self.perc_replacements = perc_replacements

    def fit(self, X, y=None):
        return self


    def transform(self, documents, labels): # TODO
        # 1. get list of words to replace with zipft law
        synonyms = self._get_words_to_replace(documents)

        # 2. get data augmentation (everytime we see word in list, we augment)
        new_documents = []
        synonyms_dict = {}
        for sentence, label in zip(documents, labels): 
            #  new_documents.append([sentence, label]) # include old sentence
            # add new sentences with synonym replacement
            words = sentence.split(' ')
            for i, word in enumerate(words):
                if word in synonyms:
                    # find list of synonym for word in not in dict
                    if word not in synonyms_dict: 
                        synonyms_dict[word] = self._get_synonyms(word)

                    # data augmentation
                    for synonym in synonyms_dict[word]: 
                        new_sentence_words = words[0:i-1] + [synonym] + words[0: i+1]\
                            if i + 1 < len(sentence) else words[0:i-1] + [synonym]
                        new_sentence = " ".join(new_sentence_words)
                        new_documents.append([new_sentence, label])


        return new_documents

    def _get_num_of_words_to_replace(self, documents): 
        """ 
        Function that return how many words to replace
        len(vocab) * perc
        """
        words = [word.lower() for sentence in complete_text.split('.') for word in sentence.split(' ')]
        return int(len(words) * self.perc_replacements)
        

    def _get_words_to_replace(self, documents): 
        """ 
        Function that get list of words to perform data augmentation on. 
        We choose the bottom from zipft law
        """
        # get word frequency count
        df_freq = self._get_word_frequency(documents)

        # get num words to find synonym
        num_words_to_replace = self._get_num_of_words_to_replace(documents)

        # find the list of words to replace
        words_to_replace = list(df_freq[:num_words_to_replace]['word'])

        return words_to_replace
        

    def _get_word_frequency(self, documents): 
        """ 
        Function that count word frequency 

        Parameters
        ----------
        documents: list of string

        Returns
        -------
        df: pandas dataframe
            dataframe with word frequency (col: word, count) in decreasing order
        """
        complete_text = " ".join(documents)
        words = [word.lower() for sentence in complete_text.split('.') for word in sentence.split(' ')]

        # remove selected words
        to_remove = [' ', ',', '', '-', '``', "'"]
        words = [word for word in words if word not in to_remove]
        
        freq = pd.value_counts(np.array(words)).rename_axis('word').reset_index(name='count')
        freq = freq.sort_values(by=['count'], ascending=True)

        return freq


    def _get_synonyms(self, word):
        """ 
        Function that return list of synonym using wordnet model

        Parameters
        ----------
        word: string
            word to find synonym for

        Returns
        -------
        synonym: list of string
            list of synonym. Include initial word
        """
        # FIXME: why does list change every time function is run

        # get all synonyms from wordnet
        synonym_tokens = wordnet.synsets(word)
        synonyms = list(set(chain.from_iterable([word.lemma_names() for word in synonym_tokens])))

        # get top synonym (current: first 5)
        top_synonyms = synonyms[:self.max_synonyms]

        # clean: replace '_' by ' '
        top_synonyms = [word.replace('_', ' ') for word in top_synonyms]

        return top_synonyms

class WordNetSynonymAugmenter(BaseEstimator, TransformerMixin):

    """ Synonym replacement using textattack WordNetAugmenter """

    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, documents, labels):
        pass
        
        
def main():
    data_augmenter = CustomSynonymReplacement()
    synonyms = data_augmenter._get_synonyms("eating")
    print(synonyms)
    #  print(data_augmenter.transform(df['word'], df['label']))
    
if __name__ == "__main__":
    main()

        
