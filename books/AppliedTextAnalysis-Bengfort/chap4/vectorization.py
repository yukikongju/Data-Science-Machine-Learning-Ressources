import nltk
import string

from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    corpus = [
        "I ate a potato yesterday.", 
        "I went to the beach a month ago.",
        "I saw a bat drinking blood at the park."
    ]
    normalized_corpus = normalize_corpus(corpus)
    bag_of_words = get_bag_of_words(corpus)

    vect = []
    for sentence in normalized_corpus:
        #  vect.append(frequency_vectorization(sentence, bag_of_words))
        vect.append(one_hot_encoding(sentence, bag_of_words))
    print(vect)

    tfidf = TfidfVectorizer()
    print(tfidf.fit_transform(corpus))




def normalize_corpus(corpus):
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
    corpus = [word_tokenize(sentence) for sentence in corpus]
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer('english')

    new_corpus = []
    for sentence in corpus:
        new_sentence = []
        for word in sentence:
            word = word.lower()
            if word not in stop_words and word not in string.punctuation:
                new_sentence.append(stemmer.stem(word))
        new_corpus.append(new_sentence)

    return new_corpus

def frequency_vectorization(sentence, bag_of_words):
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
    >>> frequency_vectorization(sentence, bag_of_words)
    >>> {'ate': 1, 'bat': 0, 'drink': 0, 'potato': 2}
    """
    #  initialize counter dict
    counter = {}
    for word in bag_of_words:
        counter[word] = 0

    # frequency count
    for word in sentence:
        counter[word] += 1

    return counter

def one_hot_encoding(sentence, bag_of_words):
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
    for word in bag_of_words:
        counter[word] = 0 

    # frequency count
    for word in sentence:
        counter[word] = 1

    return counter
    
    

    
def get_bag_of_words(corpus):
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
    corpus = [word_tokenize(sentence) for sentence in corpus]
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer('english')

    # get all different words
    bag = []
    for sentence in corpus:
        for word in sentence:
            word = word.lower()
            if word not in stop_words and word not in string.punctuation:
                bag.append(stemmer.stem(word))

    # remove dupplicates and sort by alphabetical order
    bag = list(set(bag))
    bag.sort()
    return bag



if __name__ == "__main__":
    main()


