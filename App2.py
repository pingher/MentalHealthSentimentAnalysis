#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import neattext.functions as nfx


model = joblib.load('best_model.joblib')
tfidf = joblib.load('tfidf_vectorizer.joblib')

emotion_mapping = {0: 'Anxiety 😟💭', 1: 'Normal 🙂🌤', 2: 'Depression 😔🌧', 3: 'Suicidal 💔🖤', 4: 'Stress 😫📚', 5: 'Bipolar 😄😔', 6: 'Personality disorder 🤔💔'}

def custom_preprocess(text):
    text = nfx.remove_urls(text)
    text = nfx.remove_userhandles(text)
    text = re.sub(r'[^\w\s!?]', '', text)
    text = text.lower()
    return text

st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Introduction", "EDA", "Sentiment Analyzer"])

# Introduction Page
if menu == "Introduction":
    st.title("Introduction")
    st.write("""
    Welcome to the Health Sentiment Analysis App!  
    This application helps to analyze diary entries (text) and predict the emotional sentiment behind them.
    """)

# EDA Page
elif menu == "EDA":
    st.title("Exploratory Data Analysis (EDA)")

    st.image('images/img1.jpg', caption='Distribution of Data')
    st.image('images/img2.jpg', caption='Average Character Count for each Status')
    st.image('images/img3.jpg', caption='Average Sentence Count for each Status')
    st.image('images/img4.jpg', caption='Frequency of characters : "?" and "!" across different sentiment statuses')
    st.image('images/img5.jpg', caption='Most Frequently Used Words')
    st.image('images/img6.jpg', caption='Most Frequently Used Words')
    st.image('images/img7.jpg', caption='Most Frequently Used Words')


# Project Display Page
elif menu == "Sentiment Analyzer":
    st.title("Health Sentiment Analysis")
    st.write("Analyze the sentiment of your diary entries(text).")
    
    user_input = st.text_area("Enter your diary text here:")
    
    if st.button("Analyze"):
        if user_input.strip():
            cleaned_input = custom_preprocess(user_input)
            
            processed_input = tfidf.transform([cleaned_input])
            prediction = model.predict(processed_input)[0]
            probabilities = model.predict_proba(processed_input)[0]
    
            emotion = emotion_mapping[prediction]
            confidence = np.max(probabilities)
            st.subheader(f"Predicted Emotion: {emotion}")
            st.write(f"Confidence Score: {confidence:.2f}")
    
            prob_df = pd.DataFrame({
                'Emotion': list(emotion_mapping.values()),
                'Probability': probabilities
            })
            st.bar_chart(prob_df.set_index('Emotion'))
        else:
            st.write("Please provide some input to analyze.")


# In[ ]:




