# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 12:56:57 2018

@author: bhupe
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import os
os.chdir('C:/Users/bhupe/Desktop/kaggle/Titanic ML form Disaster')


#importing dataset
test_dataset=pd.read_csv('test.csv')
train_dataset=pd.read_csv('train.csv',header=0)
Y_test=pd.read_csv('gender_submission.csv')



#preprocessing train_dataset
a=train_dataset.Age.notnull()
n=train_dataset.Embarked.notnull()
c=[]
for i in range(len(a)):
    if a[i]==False or n[i]==False:
        c.append(i)
new_dataset=train_dataset.drop(c)
X=new_dataset.iloc[:,[0,2,4,5,6,7,9,11]].values        
Y=new_dataset.iloc[:,[1]].values

#preprocessing test_dataset
a1=test_dataset.Age.notnull()
n1=test_dataset.Embarked.notnull()
c1=[]
for i in range(len(a1)):
    if a1[i]==False or n1[i]==False:
        c1.append(i)
new1_dataset=test_dataset.drop(c1)
X1=new1_dataset.iloc[:,[0,1,3,4,5,6,8,10]].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(X1[:,[6]])
X1[:,[6]] = imputer.transform(X1[:,[6]])





#preprocessing Y_test
Y_test=Y_test.drop(c1)
Y_testf=Y_test.iloc[:,[1]].values



#encoding categorical variables for new_dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,2] = labelencoder_X.fit_transform(X[:,2])
X[:,2]=X[:,2].astype(float)
X[:,7]=labelencoder_X.fit_transform(X[:,7])
X[:,7]=X[:,7].astype(float)
onehotencoder = OneHotEncoder(categorical_features = [2,7])
X = onehotencoder.fit_transform(X).toarray()



#encoding categorical variables for new1_dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X1[:,2] = labelencoder_X1.fit_transform(X1[:,2])
X1[:,2]=X1[:,2].astype(float)
X1[:,7]=labelencoder_X1.fit_transform(X1[:,7])
X1[:,7]=X1[:,7].astype(float)

onehotencoder1 = OneHotEncoder(categorical_features = [2,7],handle_unknown='ignore')
X1 = onehotencoder1.fit_transform(X1).toarray()


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X1 = sc.transform(X1)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0)
classifier.fit(X, Y)

# Predicting the Test set results
Y1 = classifier.predict(X1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_testf, Y1)
