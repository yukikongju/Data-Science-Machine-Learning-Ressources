import nltk
import string

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class OneHotVectorizer(BaseEstimator, TransformerMixin):

    """ 
    Vectorize texts using OneHotEncoding

    Examples
    --------
    >>> sentences = ['like potato sandwich banana', 'sky red rainbow']
    >>> vectorizer = OneHotVectorizer()
    >>> vectorizer.fit_transform(sentences)
    >>> [[1 1 1 0 0 1 0] [0 0 0 1 1 0 1]]
    """

    def __init__(self):
        self.vectorizer = CountVectorizer(binary=True)

    def fit(self, sentences, labels = None):
        return self

    def transform(self, sentences):
        freqs = self.vectorizer.fit_transform(sentences)
        return freqs.toarray()
        

class TextNormalizer(BaseEstimator, TransformerMixin):

    """ 
    Remove all stopwords and punctuation; lemmatize all words

    Examples
    --------
    >>> sentences = ['I like potatoes sandwiches with bananas', 'The sky is red with rainbow']
    >>> normalizer = TextNormalizer()
    >>> normalizer.fit_transform(sentences)
    >>> ['like potato sandwich banana', 'sky red rainbow']
    """

    def __init__(self, language='english'):
        self.language = language
        self.stopwords = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def _is_stop_word(self, token):
        return token.lower() in self.stopwords


    def _is_punctuation(self, token):
        return token.lower() in string.punctuation
        
    def _normalize(self, sentence):
        """ 
        Normalize sentence: remove stopwords and lemmatize
        TODO: remove punctuation
        """
        words = []
        for word in sentence.split(' '):
            if not self._is_stop_word(word):
                words.append(self.lemmatizer.lemmatize(word))
        #  return words
        return ' '.join(words) 
        

    def fit(self, x):
        return self

    def transform(self, sentences):
        """ 
        Normalize sentences
        """
        return [ self._normalize(sentence) for sentence in sentences ]
        
    
def test_wordnet_lemmatizer(text):
    lemmatizer = WordNetLemmatizer()
    for word in text.split(' '):
        print(lemmatizer.lemmatize(word))
    
def test_stopwords():
    print('a' in set(stopwords.words('english')))
    print('with' in set(stopwords.words('english')))
    
def test_vectorizer(texts):
    vectorizer = CountVectorizer(binary=True)
    freqs = vectorizer.fit_transform(texts)
    print(freqs.toarray())

def test_textnormalizer(texts):
    normalizer = TextNormalizer()
    print(normalizer.transform(texts))
    
def test_onehotvectorizer(texts):
    vectorizer = OneHotVectorizer()
    print(vectorizer.fit_transform(texts))

    
def test_pipeline(texts): 
    pipeline = Pipeline([
        ('normalizer', TextNormalizer()), 
        ('vectorizer', OneHotVectorizer())
    ])
    print(pipeline.fit_transform(texts))


if __name__ == "__main__":
    #  nltk.download('omw-1.4')
    #  nltk.download('stopwords')

    texts = [
        "I like potatoes sandwiches with bananas",
        "The sky is red with a rainbow"
    ]

    #  test_stopwords()
    #  test_wordnet_lemmatizer(texts[0])
    #  test_textnormalizer(texts)
    #  test_vectorizer(texts)
    test_pipeline(texts)
