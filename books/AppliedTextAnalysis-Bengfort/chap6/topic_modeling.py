import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.base import BaseEstimator, TransformerMixin

from utils import OneHotVectorizer, TextNormalizer


class TopicModelling(BaseEstimator, TransformerMixin):

    def __init__(self, num_topics):
        self.num_topics = num_topics
        self.model = Pipeline([
            ('normalizer', TextNormalizer()),
            ('vectorizer', OneHotVectorizer),
            ('model', LatentDirichletAllocation(n_components=self.num_topics))
        ])

    #  def fit(self, sentences, labels=None):
    #      return self.model

    #  def transform(self, sentences):
    #      return self.model.fit_transform(sentences)
        
    def fit_transform(self, sentences):
        self.model.fit_transform(sentences)

        return self.model
        


    def get_topics(self, n):
        """ 
        n: int 
            num of top terms by topics
        """
        vectorizer = self.model.named_steps['vect']
        model = self.model.steps[-1][1]
        names = vectorizer.get_feature_names()
        topics = dict()
        for idx, topic in enumerate(model.components_):
            features = topic.argsort()[:-(n - 1): -1]
            tokens = [names[i] for i in features]
            topics[idx] = tokens
        return topics
    
        
def test_lda(X):
    lda = LatentDirichletAllocation(25)
    lda.fit_transform(X)
    outputs = lda.transform(X)
    print(res)
    

        

def test_topic_modelling(sentences):
    lda = TopicModelling(num_topics=25)
    res = lda.fit_transform(sentences)
    #  topics = lda.get_topics(10)
    #  print(topics)
    


if __name__ == "__main__":
    filepath = 'data/headlines/headlines_subset.csv'
    sentences = pd.read_csv(filepath)['headline_text']

    pipeline = Pipeline([
        ('normalizer', TextNormalizer()),
        ('vectorizer', OneHotVectorizer())
    ])
    #  X = pipeline.fit_transform(sentences)

    #  test_lda(X)
    test_topic_modelling(sentences)



