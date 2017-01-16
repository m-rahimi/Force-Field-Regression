#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 11:49:09 2016

@author: amin
"""

# Import libraries
import numpy as np
import pandas as pd
import os
from time import time

os.chdir("/home/amin/Desktop/Udacity-Machine_Learning/ForceField")
dir = os.getcwd()
print "Path at terminal when executing this file:\n{}".format(dir)

# Read data
data = pd.read_csv("FF.csv")
print "Student data read successfully!"
data.describe()
data.shape

#Preparaing the data
feature_cols = list(data.columns[5:15])
feature_cols.remove('alpha')
feature_cols.remove('beta')
feature_cols.remove('gamma')
feature_cols.remove('a')
feature_cols.remove('b')
feature_cols.remove('c')
target_cols = list(data.columns[1:5])

# Show the list of columns
print "Feature columns:\n{}".format(feature_cols)
print "\nTarget columns: {}".format(target_cols)

X_all = data[feature_cols]
y_all = data[target_cols]

#split the data
from sklearn.cross_validation import train_test_split
num_test = int(X_all.shape[0] * 0.1)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=123)

# Linear regression
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
reg.score(X_train, y_train)

# evaluate the model
from sklearn.metrics import r2_score
y_pred = reg.predict(X_test)
score = r2_score(y_test,y_pred, multioutput='raw_values')
print "Model has a coefficient of determination, R^2, of {}.".format(score)

score = r2_score(y_test,y_pred, multioutput='variance_weighted')
print "Model has a coefficient of determination, R^2, of {}.".format(score)

score = r2_score(y_test,y_pred, multioutput='uniform_average')
print "Model has a coefficient of determination, R^2, of {}.".format(score)

# MSE
from sklearn.metrics import mean_squared_error
msq = mean_squared_error(y_test, y_pred, multioutput='raw_values')
print "Model has a coefficient of determination, MSQ, of {}.".format(msq)

msq = mean_squared_error(y_test, y_pred, multioutput='uniform_average')
print "Model has a coefficient of determination, MSQ, of {}.".format(msq)