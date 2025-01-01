#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score
import neattext.functions as nfx
import joblib


# In[2]:


df = pd.read_csv('emotion_dataset.csv', index_col=0)


# ### Data Cleaning

# In[3]:


df = df.dropna()
df = df.drop_duplicates()


# ### Data Visualization (5 graphs) 

# In[4]:


import matplotlib.pyplot as plt
df2 = df['status'].value_counts()
plt.figure(figsize=(11,3))
plt.bar(df2.index,df2.values)
plt.xlabel('Status')
plt.ylabel('Total Amount')


# In[5]:


df2 = df.copy() 
df2['char_count'] = df2['statement'].apply(len)
avg_char_count = df2.groupby('status')['char_count'].mean().sort_values()

plt.figure(figsize=(11,3))
plt.ylabel('Average Character Count')
plt.xlabel('Statuses')
plt.bar(avg_char_count.index, avg_char_count.values)
plt.title('Average Character Count Across Status')
plt.show()


# In[6]:


df2['sentence_count'] = df2['statement'].apply(
    lambda x: max(1, x.count('.') + x.count('!') + x.count('?'))
)
avg_sent_count = df2.groupby('status')['sentence_count'].mean().sort_values()
plt.figure(figsize=(11,3))
plt.ylabel('Average Sentence Count')
plt.xlabel('Statuses')
plt.bar(avg_sent_count.index, avg_sent_count.values)
plt.title('Average Sentence Count for Statements in Status')
plt.show()


# In[7]:


df2['question_marks'] = df2['statement'].apply(lambda x: x.count('?'))
df2['exclamation_marks'] = df2['statement'].apply(lambda x: x.count('!'))

question_mark_count = df2.groupby('status')['question_marks'].sum()
exclamation_mark_count = df2.groupby('status')['exclamation_marks'].sum()

plt.figure(figsize=(10, 6))

plt.subplot(2,1,1)
plt.bar(question_mark_count.index, question_mark_count.values, color='skyblue')
plt.title("Frequency of '?' Across Status")
plt.xlabel("Status")
plt.ylabel("Count of '?'")
plt.tight_layout()

plt.subplot(2,1,2)
plt.bar(exclamation_mark_count.index, exclamation_mark_count.values, color='salmon')
plt.title("Frequency of '!' Across Status")
plt.xlabel("Status")
plt.ylabel("Count of '!'")
plt.tight_layout()


# In[8]:


def custom_preprocess(text):
    text = nfx.remove_urls(text)
    text = nfx.remove_userhandles(text)
    text = re.sub(r'[^\w\s!?]', '', text)
    text = text.lower()
    return text

df['cleaned_text'] = df['statement'].apply(custom_preprocess)


# In[9]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

statuses = df['status'].unique()
fig, axes = plt.subplots(len(statuses), 1, figsize=(10, 20))

for i, status in enumerate(statuses):
    text = " ".join(df[df['status'] == status]['cleaned_text'])
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    axes[i].imshow(wordcloud, interpolation='bilinear')
    axes[i].set_title(f"Word Cloud for {status}", fontsize=14)
    axes[i].axis('off')

plt.tight_layout()
plt.show()


# In[10]:


emotion_mapping = {label: idx for idx, label in enumerate(df['status'].unique())}
df['final_status'] = df['status'].map(emotion_mapping)


# In[12]:


X = df['cleaned_text']

y = df['final_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[13]:


tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_test_tfidf = tfidf.transform(X_test).toarray()


# In[14]:


models = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression()
}

best_model = None
best_accuracy = 0
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_tfidf, y_train)
    
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.2f}")
    
    if accuracy > best_accuracy:
        best_model = model
        best_accuracy = accuracy

y_pred = best_model.predict(X_test_tfidf)
print("Best Model Performance:")
print(classification_report(y_test, y_pred))


joblib.dump(best_model, 'best_model.joblib')
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')

