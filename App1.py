#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
import re
from bs4 import BeautifulSoup

# Load models
rf_model = joblib.load('random_forest_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')

# Load CountVectorizer
cv = joblib.load('count_vectorizer.pkl')

# Define preprocessing function
def preprocess(q):
    q = str(q).lower().strip()
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    q = q.replace('[math]', '')
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)
    contractions = {"ain't": "am not", "aren't": "are not", "can't": "can not", "can't've": "can not have",
                    "'cause": "because", "could've": "could have", "couldn't": "could not", "couldn't've": "could not have",
                    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hadn't've": "had not have",
                    "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'd've": "he would have", "he'll": "he will",
                    "he'll've": "he will have", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
                    "how's": "how is", "i'd": "i would", "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have",
                    "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
                    "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
                    "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
                    "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                    "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                    "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
                    "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not",
                    "shouldn't've": "should not have", "so've": "so have", "so's": "so as", "that'd": "that would", "that'd've": "that would have",
                    "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "they'd": "they would",
                    "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have",
                    "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                    "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                    "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                    "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is",
                    "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have",
                    "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have",
                    "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                    "you'll've": "you will have", "you're": "you are", "you've": "you have"}

    q_decontracted = []

    for word in q.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")

    q = BeautifulSoup(q)
    q = q.get_text()

    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()

    return q

def extract_features(q1, q2):
    input_data = pd.DataFrame([[q1, q2]], columns=['question1', 'question2'])
    input_data['question1'] = input_data['question1'].apply(preprocess)
    input_data['question2'] = input_data['question2'].apply(preprocess)
    
    q1_arr = cv.transform(input_data['question1']).toarray()
    q2_arr = cv.transform(input_data['question2']).toarray()
    
    token_features = input_data.apply(fetch_token_features, axis=1)
    length_features = input_data.apply(fetch_length_features, axis=1)
    fuzzy_features = input_data.apply(fetch_fuzzy_features, axis=1)
    
    final_features = np.hstack((q1_arr, q2_arr, token_features, length_features, fuzzy_features))
    
    return final_features

# Streamlit app
st.title('Duplicate Question Detection')

question1 = st.text_area('Question 1')
question2 = st.text_area('Question 2')

if st.button('Predict'):
    features = extract_features(question1, question2)
    
    rf_prediction = rf_model.predict(features)[0]
    xgb_prediction = xgb_model.predict(features)[0]
    
    st.write(f'Random Forest Prediction: {"Duplicate" if rf_prediction else "Not Duplicate"}')
    st.write(f'XGBoost Prediction: {"Duplicate" if xgb_prediction else "Not Duplicate"}')


# In[ ]:




