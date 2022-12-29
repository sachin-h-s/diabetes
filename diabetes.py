# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 15:25:49 2021

@author: sachin h s
"""

import pandas as pd
import numpy as np
diabetes = pd.read_csv(r'C:\Users\sachin h s\Downloads\inventa\DATA SCIENCE data sets\diabetes.csv')
X=diabetes.iloc[:,:-1]
y=diabetes.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)
import pickle
pickle_out = open("classifier.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()