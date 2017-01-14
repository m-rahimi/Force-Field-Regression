# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 16:54:51 2016

@author: Amin
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
target_cols = list(data.columns[1:5])

# Show the list of columns
print "Feature columns:\n{}".format(feature_cols)
print "\nTarget columns: {}".format(target_cols)

X_all = data[feature_cols]
y_all = data[target_cols]

# get the feature correlations
corr = X_all.corr()

# remove first row and last column for a cleaner look
corr.drop(['a'], axis=0, inplace=True)
corr.drop(['Solvation'], axis=1, inplace=True)

# create a mask so we only see the correlation values once
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, 1)] = True

# plot the heatmap
import matplotlib.pyplot as plt
import seaborn as sns
with sns.axes_style("white"):
    sns.heatmap(abs(corr), mask=mask, annot=True, cmap='RdBu', fmt='+.2f', cbar=False)
    
# density is highly correlated to the unit cell size and we can remove them from features
feature_cols.remove('a')
feature_cols.remove('b(A)')
feature_cols.remove('c(A)')

X_all = data[feature_cols]

axes = pd.scatter_matrix(X_all, alpha = 0.3, figsize = (14,8), diagonal = 'kde')
corr = X_all.corr().as_matrix()
for i, j in zip(*np.triu_indices_from(axes, k=1)):
    axes[i, j].annotate("%.3f" %abs(corr[i,j]), (0.8, 0.8), xycoords='axes fraction', ha='center', va='center')
    
###################
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.1)
fig.suptitle('density and target variables', fontsize=20)

# Make first subplot
dt = data[['sigP', 'epsiP', 'density']]
dt2 = dt.pivot_table(index='sigP', columns='epsiP', values='density')

X=dt.columns.values
Y=dt.index.values
Z=dt.values
x,y=np.meshgrid(X, Y)

cp1 = axs[0, 0].contourf(x, y, Z, alpha=0.7, cmap=plt.cm.jet)
axs[0, 0].set_xlabel('epsiP', fontsize=15)
axs[0, 0].set_ylabel('sigP', fontsize=15)
fig.colorbar(cp1, ax=axs[0,0])

# Make second subplot
dt = data[['sigO', 'epsiO', 'density']]
dt = dt.pivot_table(index='sigO', columns='epsiO', values='density')

X=dt.columns.values
Y=dt.index.values
Z=dt.values
x,y=np.meshgrid(X, Y)

cp1 = axs[0, 1].contourf(x, y, Z, alpha=0.7, cmap=plt.cm.jet)
axs[0, 1].set_xlabel('epsiO', fontsize=15)
axs[0, 1].set_ylabel('sigO', fontsize=15)
fig.colorbar(cp1, ax=axs[0,1])

# Make 3th subplot
dt = data[['sigP', 'sigO', 'density']]
dt = dt.pivot_table(index='sigP', columns='sigO', values='density')

X=dt.columns.values
Y=dt.index.values
Z=dt.values
x,y=np.meshgrid(X, Y)

cp1 = axs[1, 0].contourf(x, y, Z, alpha=0.7, cmap=plt.cm.jet)
axs[1, 0].set_xlabel('sigO', fontsize=15)
axs[1, 0].set_ylabel('sigP', fontsize=15)
fig.colorbar(cp1, ax=axs[1,0])

# Make 3th subplot
dt = data[['epsiP', 'epsiO', 'density']]
dt = dt.pivot_table(index='epsiP', columns='epsiO', values='density')

X=dt.columns.values
Y=dt.index.values
Z=dt.values
x,y=np.meshgrid(X, Y)

cp1 = axs[1, 1].contourf(x, y, Z, alpha=0.7, cmap=plt.cm.jet)
axs[1, 1].set_xlabel('epsiO', fontsize=15)
axs[1, 1].set_ylabel('epsiP', fontsize=15)
fig.colorbar(cp1, ax=axs[1,1])

plt.show()

