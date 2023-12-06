# -*- coding: utf-8 -*-

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.feature_extraction.text import CountVectorizer


# loading the saved models

bagging_model = pickle.load(open('C:/Users/shrut/OneDrive/Desktop/CC/IML_PROJECT/Model/MLmodel.sav', 'rb'))

vect = pickle.load(open('C:/Users/shrut/OneDrive/Desktop/CC/IML_PROJECT/vect.pkl', 'rb'))  


# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Sentiment analysis',                          
                          ['Sentiment Prediction'],                        
                          default_index=0)
    
    
# Diabetes Prediction Page
if (selected == 'Sentiment Prediction'):
    
    # page title
    st.title('Sentiment analysis using machine learning')
    tweet = st.text_input('Text')
    tweet_new = vect.transform([tweet])
    
    # code for Prediction
    pred = ''
    
    # creating a button for Prediction
    
    if st.button('Sentiment analysis result'):
        prediction = bagging_model.predict(tweet_new)
        
        if (prediction[0] == 1):
          pred = 'Positive'
        else:
          pred = 'Negative'
        
    st.success(pred)


