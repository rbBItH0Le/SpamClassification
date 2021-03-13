# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 21:43:17 2021

@author: rohan
"""
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
import numpy as np
df=pd.read_csv('smsspamcollection.tsv',sep='\t')
wn=WordNetLemmatizer()
corpus=[]
for i in range(len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['message'][i])
    review = review.lower()
    review = review.split()  
    review = [wn.lemmatize(word)for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
X=tf.fit_transform(corpus).toarray()

y=pd.get_dummies(df['label'],drop_first=True)

from sklearn.naive_bayes import MultinomialNB
mb=MultinomialNB()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

mb.fit(X_train,y_train)
pred=mb.predict(X_test)

from sklearn.metrics import accuracy_score
sc=accuracy_score(pred,y_test)