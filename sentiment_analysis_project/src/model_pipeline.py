from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import spacy
import re

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass  # no hyperparameters needed for this step

    def fit(self, X, y=None):
        return self  # no fitting needed

    def transform(self, X):
        return X.apply(self.text_cleaning)  # apply text cleaning function to each row

    def text_cleaning(self, text):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        cleaned_text = ' '.join([token.lemma_ for token in doc])  # lemmatization
        cleaned_text = re.sub(r'([^\s\w]|_)+', ' ', cleaned_text)  # keeps only letters and numbers
        return re.sub(r'\s+', ' ', cleaned_text).strip()  # removes extra spaces


def pipeline_function(model):
    return Pipeline([
        ('text_cleaner', TextCleaner()),  # custom text cleaning transformer
        ('vectorizer', TfidfVectorizer(max_features=500)),  # convert text to numerical features using tf-idf
        ('classifier', model)  # classifier model
    ])
