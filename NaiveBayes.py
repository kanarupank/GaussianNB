#!/usr/bin/env python
# coding: utf-8

# In[1]:


print('Naive Bayes on Wisconsin Brest Cancer Data')


# In[16]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set()

#column names
col_names = ['Code Number', 'Clump Thickness','Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

# load dataset
#wbcd = pd.read_csv('wbcd.csv', header=None, names=col_names)
wbcdReplacedData = pd.read_csv('wbcdReplacedData.csv', header=None, names=col_names)
feature_cols = [ 'Clump Thickness','Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
features= wbcdReplacedData[feature_cols] # Features
result = wbcdReplacedData.Class # Target variable
featuresReplacedData= wbcdReplacedData[feature_cols] # Features all data
resultReplacedData = wbcdReplacedData.Class # Target variable all data


X_train, X_test, y_train, y_test = train_test_split(featuresReplacedData, resultReplacedData, test_size = 0.35)


#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

