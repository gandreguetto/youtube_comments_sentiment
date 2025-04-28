from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import spacy
import re

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(self.text_cleaning)

    def text_cleaning(self, text):
        doc = self.nlp(text)
        cleaned_text = ' '.join([token.lemma_ for token in doc])
        cleaned_text = re.sub(r'([^\s\w]|_)+', ' ', cleaned_text)
        return re.sub(r'\s+', ' ', cleaned_text).strip()

def pipeline_function(model):
    return Pipeline([
        ('text_cleaner', TextCleaner()),  # custom text cleaning transformer
        ('vectorizer', TfidfVectorizer(max_features=500)),  # convert text to numerical features using tf-idf
        ('classifier', model)  # classifier model
    ])
