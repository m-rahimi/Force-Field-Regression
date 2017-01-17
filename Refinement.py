#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 16:16:15 2016

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

# Preprocessing
from sklearn import preprocessing
X_scale = preprocessing.scale(X_all)
y_scale = preprocessing.scale(y_all)

#split the data
from sklearn.cross_validation import train_test_split
num_test = int(X_all.shape[0] * 0.1)
#X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_scale, test_size=num_test, random_state=123)

# MultiTaskElasticNet
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import grid_search
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score

model = make_pipeline(PolynomialFeatures(), linear_model.ElasticNet(max_iter=2000))
parameters = {'elasticnet__alpha' : map(lambda x: x/10.0, range(1,10)), 
              'elasticnet__l1_ratio' : map(lambda x: x/10.0, range(1,10)),
              'polynomialfeatures__degree' : [1, 2, 3], 
              'polynomialfeatures__interaction_only' :[True, False]}


r2_scorer = make_scorer(r2_score, multioutput='uniform_average')
grid = grid_search.GridSearchCV(estimator=model, param_grid=parameters, scoring=r2_scorer, cv=5)
grid_obj = grid.fit(X_train, y_train)

grid_best = grid_obj.best_estimator_
print grid_best

grid_best.fit(X_train, y_train)
y_pred = grid_best.predict(X_test)
score = r2_score(y_test,y_pred, multioutput='raw_values')
print "Full set of scores {}.".format(score)
score = r2_score(y_test,y_pred, multioutput='uniform_average')
print "Model has a coefficient of determination, R^2, of {}.".format(score)