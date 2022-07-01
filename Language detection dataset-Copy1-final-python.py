#!/usr/bin/env python
# coding: utf-8

# # IMPORTING DATASET


from pyngrok import ngrok
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")


data = pd.read_csv("Languages.csv")
data.head(20)


data.tail(20)


data["language"].value_counts()


data.isna().sum()


data.dropna(inplace=True)


X = data["text"]
y = data["language"]


le = LabelEncoder()
y = le.fit_transform(y)


# creating a list for appending the preprocessed text
data_list = []
# iterating through all the text
for text in X:
    #         print(count,text)
    # removing the symbols and numbers
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    # converting the text to lower case
    text = text.lower()
    # appending to data_list
    data_list.append(text)


cv = CountVectorizer()
X = cv.fit_transform(data_list).toarray()
X.shape  # (10337, 39419)


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


model = MultinomialNB()
model.fit(x_train, y_train)


y_pred = model.predict(x_test)


ac = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy is :", ac)
# Accuracy is : 0.9772727272727273


plt.figure(figsize=(15, 10))
sns.heatmap(cm, annot=True)
plt.show()


def predict(text):
    # converting text to bag of words model (Vector)
    x = cv.transform([text]).toarray()
    lang = model.predict(x)  # predicting the language
    # finding the language corresponding the the predicted value
    lang = le.inverse_transform(lang)
    print("The langauge is", lang[0])  # printing the language


predict("Natural Language Processing is essential to analyze text data")


predict("भारतस्य पूर्वभागे बाङ्ग्लादेशः बर्मादेशः, बङ्")


pred = predict("முலாயம் சிங் மருத்துவமனையில் அனுமதி")


# pip install save


# pred.save("modelname")


# pip install streamlit -q


# get_ipython().run_cell_magic('writefile', 'app.py',
#                              "import streamlit as st\nst.write('#Language Prediction')")


# pip install pyngrok


# :


# streamlit run app.py & npx localtunnel - -port 8501


# get_ipython().system('pip install streamlit')
# npm install localtunnel
# get_ipython().system('streamlit run http://app.py &>/dev/null&')
# get_ipython().system('npx localtunnel --port 8501')

# install local tunnel
