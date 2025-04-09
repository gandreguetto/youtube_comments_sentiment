import pickle
from sklearn.base import BaseEstimator, TransformerMixin
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")
import re

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass  # no hyperparameters needed for this step

    def fit(self, X, y=None):
        return self  # no fitting needed

    def transform(self, X):
        return X.apply(self.text_cleaning)  # apply text cleaning function to each row

    def text_cleaning(self, text):
        doc = nlp(text)
        cleaned_text = ' '.join([token.lemma_ for token in doc])  # lemmatization
        cleaned_text = re.sub(r'([^\s\w]|_)+', ' ', cleaned_text)  # keeps only letters and numbers
        return re.sub(r'\s+', ' ', cleaned_text).strip()  # removes extra spaces

df = 2

model = pickle.load(open('sentiment-analysis-logreg.pkl', 'rb'))

def main(): 
    st.title("Sentiment Analysis")

    comment = st.text_input("Comment","")

    if st.button("Predict"):

        prediction = model.predict(pd.Series(comment))

        if prediction == 1:
            st.badge('This is a positive comment!')
        else:
            st.badge('This is a negative comment!')

if __name__=='__main__': 
    main()
