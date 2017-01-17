#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 16:12:54 2016

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
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=123)

# Linear regression
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
reg.score(X_train, y_train)

# evaluate the linear model
from sklearn.metrics import r2_score
y_pred = reg.predict(X_test)
score = r2_score(y_test,y_pred, multioutput='raw_values')
print "Model has four coefficients of determination, R^2, of {}.".format(score)

score_t = r2_score(y_test,y_pred, multioutput='uniform_average')
print "The average of four coefficients of determination, R^2, of {}.".format(score_t)

#record the score for linear regression
result = {'linear': np.append(score, score_t)}

# Ridge model
from sklearn.linear_model import Ridge
reg = Ridge()

# Gridsearch 
from sklearn import grid_search
from sklearn.metrics import make_scorer

parameters = {'alpha' : map(lambda x: x/10.0, range(0,10))}

r2_scorer = make_scorer(r2_score, multioutput='uniform_average')

grid = grid_search.GridSearchCV(estimator=reg, param_grid=parameters, scoring=r2_scorer, cv=5)
grid_obj = grid.fit(X_train, y_train)

grid_best = grid_obj.best_estimator_
print grid_best

grid_best.fit(X_train, y_train)
y_pred = grid_best.predict(X_test)
score = r2_score(y_test,y_pred, multioutput='raw_values')
print "Full set of scores {}.".format(score)
score_t = r2_score(y_test,y_pred, multioutput='uniform_average')
print "Model has a coefficient of determination, R^2, of {}.".format(score_t)

#record the score for linear regression
result.update({'ridge': np.append(score, score_t)})


# Lasso model
from sklearn.linear_model import Lasso
reg = Lasso()

parameters = {'alpha' : map(lambda x: x/10.0, range(1,10))}

r2_scorer = make_scorer(r2_score, multioutput='uniform_average')

grid = grid_search.GridSearchCV(estimator=reg, param_grid=parameters, scoring=r2_scorer, cv=5)
grid_obj = grid.fit(X_train, y_train)

grid_best = grid_obj.best_estimator_
print grid_best

grid_best.fit(X_train, y_train)
y_pred = grid_best.predict(X_test)
score = r2_score(y_test,y_pred, multioutput='raw_values')
print "Full set of scores {}.".format(score)
score_t = r2_score(y_test,y_pred, multioutput='uniform_average')
print "Model has a coefficient of determination, R^2, of {}.".format(score_t)

#record the score for linear regression
result.update({'lasso': np.append(score, score_t)})

# polynomial + linear regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
reg = make_pipeline(PolynomialFeatures(), linear_model.LinearRegression())

parameters = {'polynomialfeatures__degree' : [1, 2, 3], 
              'polynomialfeatures__interaction_only' :[True, False]}

          
r2_scorer = make_scorer(r2_score, multioutput='variance_weighted')

grid = grid_search.GridSearchCV(estimator=reg, param_grid=parameters, scoring=r2_scorer, cv=5)
grid_obj = grid.fit(X_train, y_train)

grid_best = grid_obj.best_estimator_
print grid_best

grid_best.fit(X_train, y_train)
y_pred = grid_best.predict(X_test)
score = r2_score(y_test,y_pred, multioutput='raw_values')
print "Full set of scores {}.".format(score)
score_t = r2_score(y_test,y_pred, multioutput='uniform_average')
print "Model has a coefficient of determination, R^2, of {}.".format(score_t)

#record the score for linear regression
result.update({'pol_linear': np.append(score, score_t)})

# polynomial + Ridge
reg = make_pipeline(PolynomialFeatures(), Ridge())

parameters = {'ridge__alpha' : map(lambda x: x/10.0, range(1,10)),
              'polynomialfeatures__degree' : [1, 2, 3], 
              'polynomialfeatures__interaction_only' :[True, False]}

r2_scorer = make_scorer(r2_score, multioutput='uniform_average')

grid = grid_search.GridSearchCV(estimator=reg, param_grid=parameters, scoring=r2_scorer, cv=5)
grid_obj = grid.fit(X_train, y_train)

grid_best = grid_obj.best_estimator_
print grid_best

grid_best.fit(X_train, y_train)
y_pred = grid_best.predict(X_test)
score = r2_score(y_test,y_pred, multioutput='raw_values')
print "Full set of scores {}.".format(score)
score_t = r2_score(y_test,y_pred, multioutput='uniform_average')
print "Model has a coefficient of determination, R^2, of {}.".format(score_t)   

#record the score for linear regression
result.update({'pol_ridge': np.append(score, score_t)})    

# polynomial + Lasso
reg = make_pipeline(PolynomialFeatures(), Lasso(max_iter=2000))

parameters = {'lasso__alpha' : map(lambda x: x/10.0, range(1,10)),
              'polynomialfeatures__degree' : [1, 2, 3], 
              'polynomialfeatures__interaction_only' :[True, False]}

r2_scorer = make_scorer(r2_score, multioutput='uniform_average')

grid = grid_search.GridSearchCV(estimator=reg, param_grid=parameters, scoring=r2_scorer, cv=5)
grid_obj = grid.fit(X_train, y_train)

grid_best = grid_obj.best_estimator_
print grid_best

grid_best.fit(X_train, y_train)
y_pred = grid_best.predict(X_test)
score = r2_score(y_test,y_pred, multioutput='raw_values')
print "Full set of scores {}.".format(score)
score_t = r2_score(y_test,y_pred, multioutput='uniform_average')
print "Model has a coefficient of determination, R^2, of {}.".format(score_t)  
        
#record the score for linear regression
result.update({'pol_lasso': np.append(score, score_t)})


# GBoosting
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
model = MultiOutputRegressor(GradientBoostingRegressor())

parameters = {'estimator__max_depth' : [2, 3, 4, 5, 6, 7], 'estimator__learning_rate' : [0.1, 0.2, 0.3], 
              'estimator__n_estimators' :[100, 200, 300], 'estimator__max_features':[1, 2, 3, 4]}
r2_scorer = make_scorer(r2_score, multioutput='uniform_average')
grid = grid_search.GridSearchCV(estimator=model, param_grid=parameters, scoring=r2_scorer, cv=5)
grid_obj = grid.fit(X_train, y_train)

grid_best = grid_obj.best_estimator_
print grid_best

grid_best.fit(X_train, y_train)
y_pred = grid_best.predict(X_test)
score = r2_score(y_test,y_pred, multioutput='raw_values')
print "Full set of scores {}.".format(score)
score_t = r2_score(y_test,y_pred, multioutput='uniform_average')
print "Model has a coefficient of determination, R^2, of {}.".format(score_t)

result.update({'GBR': np.append(score, score_t)})


#############################################################################
# result of ElasticNet
result.update({'pol_elastic':np.array( [0.91859219, -0.00554541,  0.96409356,  0.32188378, 0.5497560308])})

pd_result = pd.DataFrame(result)
pd_result = pd_result.drop('pol_linear', axis=1)
pd_result = pd_result.drop('pol_ridge', axis=1)

pd_result = pd_result[['linear', 'ridge', 'lasso', 'pol_lasso', 'pol_elastic', 'GBR']]

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize = (14,8))
pd_result.plot.bar(ax = ax)
ax.set_ylabel("r2 score")
label = ['sigP', 'epsiP', 'sigO', 'epsiO', 'average']
ax.set_xticklabels(label, rotation=0)