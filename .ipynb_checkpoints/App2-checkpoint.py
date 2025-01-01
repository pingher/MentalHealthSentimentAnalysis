#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import neattext.functions as nfx
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


model = joblib.load('best_model.joblib')
tfidf = joblib.load('tfidf_vectorizer.joblib')

# Emotion mapping
emotion_mapping = {0: 'Anxiety', 1: 'Normal', 2: 'Depression', 3: 'Suicidal', 4: 'Stress', 5: 'Bipolar', 6: 'Personality disorder'}

# Preprocessing function
def custom_preprocess(text):
    text = nfx.remove_urls(text)
    text = nfx.remove_userhandles(text)
    text = re.sub(r'[^\w\s!?]', '', text)
    text = text.lower()
    return text

# OCR function for image input
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    print(text)
    return text

# Streamlit app UI
st.title("Health Sentiment Analysis")
st.write("Analyze the sentiment of your diary entries.")

# Input options: text or image
option = st.radio("Select input method:", ("Enter Text", "Upload Image"))

if option == "Enter Text":
    user_input = st.text_area("Enter your diary text here:")
elif option == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image containing text:", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        user_input = extract_text_from_image(image)
        st.write("Extracted Text:")
        st.write(user_input)
    else:
        user_input = ""

if st.button("Analyze"):
    if user_input.strip():
        cleaned_input = custom_preprocess(user_input)
        st.write(cleaned_input)
        
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




