#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 19:16:49 2019

@author: Will
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

#Take care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Encording categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[: , 0] = labelencoder_X.fit_transform(X[: , 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
X[: , 0] = labelencoder_X.fit_transform(X[: , 0])
Y = labelencoder_X.fit_transform(Y)

#splitting the dataset into the training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_Test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)