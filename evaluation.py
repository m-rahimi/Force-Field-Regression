#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 16:38:17 2016

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

# Create 10 cross-validation sets for training and testing
from sklearn.cross_validation import ShuffleSplit, train_test_split
cv = ShuffleSplit(X_scale.shape[0], n_iter = 10, test_size = 0.1, random_state = 123)

# Generate the training set sizes 
train_sizes = np.rint(np.linspace(X_scale.shape[0]*0.1, X_scale.shape[0]*0.9-1, 9)).astype(int)

# MultiTaskElasticNet
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

regressor = make_pipeline(PolynomialFeatures(degree=3, include_bias=True, interaction_only=False) \
                          , linear_model.MultiTaskElasticNet(alpha=0.0, copy_X=True, fit_intercept=True, l1_ratio=0.1,
          max_iter=2000, normalize=False, random_state=None,
          selection='cyclic', tol=0.0001, warm_start=False))

# Calculate the training and testing scores
import sklearn.learning_curve as curves
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
r2_scorer = make_scorer(r2_score, multioutput='uniform_average')
sizes, train_scores, test_scores = curves.learning_curve(regressor, X_all, y_scale, \
            cv = cv, train_sizes = train_sizes, scoring = r2_scorer)

# Find the mean and standard deviation for smoothing
train_std = np.std(train_scores, axis = 1)
train_mean = np.mean(train_scores, axis = 1)
test_std = np.std(test_scores, axis = 1)
test_mean = np.mean(test_scores, axis = 1)


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize = (14,8))

ax.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')
ax.plot(sizes, test_mean, 'o-', color = 'g', label = 'Testing Score')
ax.fill_between(sizes, train_mean - train_std, \
                train_mean + train_std, alpha = 0.15, color = 'r')
ax.fill_between(sizes, test_mean - test_std, \
                test_mean + test_std, alpha = 0.15, color = 'g')
ax.legend()      
# Labels
ax.set_title('ElasticNet Regressor', fontsize=20)
ax.set_xlabel('Number of Training data')
ax.set_ylabel('Score')
ax.set_xlim([400, X_all.shape[0]*0.9])
#ax.set_ylim([-0.05, 1.05])
    
# GBR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
regressor = MultiOutputRegressor(GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=6, max_features=4,
             max_leaf_nodes=None, min_impurity_split=1e-07,
             min_samples_leaf=1, min_samples_split=2,
             min_weight_fraction_leaf=0.0, n_estimators=100,
             presort='auto', random_state=None, subsample=1.0, verbose=0,
             warm_start=False))

sizes, train_scores, test_scores = curves.learning_curve(regressor, X_all, y_scale, \
            cv = cv, train_sizes = train_sizes, scoring = r2_scorer)

# Find the mean and standard deviation for smoothing
train_std = np.std(train_scores, axis = 1)
train_mean = np.mean(train_scores, axis = 1)
test_std = np.std(test_scores, axis = 1)
test_mean = np.mean(test_scores, axis = 1)


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize = (14,8))

ax.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')
ax.plot(sizes, test_mean, 'o-', color = 'g', label = 'Testing Score')
ax.fill_between(sizes, train_mean - train_std, \
                train_mean + train_std, alpha = 0.15, color = 'r')
ax.fill_between(sizes, test_mean - test_std, \
                test_mean + test_std, alpha = 0.15, color = 'g')
ax.legend()      
# Labels
ax.set_title('Gradent Boosting Regressor', fontsize=20)
ax.set_xlabel('Number of Training data')
ax.set_ylabel('Score')
ax.set_xlim([400, X_all.shape[0]*0.9])