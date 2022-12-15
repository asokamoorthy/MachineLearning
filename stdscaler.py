# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:21:25 2018

@author: Asokamoorthy K
"""

http://benalexkeen.com/feature-scaling-with-scikit-learn/


from sklearn import preprocessing
import numpy as np

x = np.array([[-100],
              [-400],
              [22],
              [100999],
              [0]])

stdscaler = preprocessing.StandardScaler()

X1 = stdscaler.fit_transform(x)
print(X1)

"""

Standardize Data
Standardization is a useful technique to transform attributes with a 
Gaussian distribution and diﬀering means and standard 
deviations to a standard Gaussian distribution with a 
mean of 0 and a standard deviation of 1. 
It is most suitable for techniques that assume a Gaussian distribution 
in the input variables and work better with rescaled data, 
such as linear regression, logistic regression and 
linear discriminate analysis.
You can standardize data using scikit-learn with the 
StandardScaler

"""
# Standardize data (0 mean, 1 stdev) 
from sklearn.preprocessing import StandardScaler 
from pandas import read_csv 
from numpy import set_printoptions 

filename = 'prima-indians-diabetes.data.csv' 
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] 
dataframe = read_csv(filename, names=names) 
array = dataframe.values
 # separate array into input and output components 
X = array[:,0:4] 
Y = array[:,8] 
scaler = StandardScaler().fit(X) 
rescaledX = scaler.transform(X) 

# summarize transformed data 
set_printoptions(precision=3) 
print(rescaledX[0:5,:])
####################################################


# Feature Extraction with Univariate
# Statistical Tests (Chi-squared for classification)

from pandas import read_csv
from numpy import set_printoptions 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# load data
filename = 'prima-indians-diabetes.data.csv' 
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] 
dataframe = read_csv(filename, names=names) 
array = dataframe.values 
X = array[:,0:4]
Y = array[:,8]
# feature extraction
#highest scores 
test = SelectKBest(score_func=chi2, k=4) 
fit = test.fit(X, Y) 
# summarize scores 
set_printoptions(precision=3) 
print(fit.scores_) 

features = fit.transform(X) 
features
# summarize selected features
print(features[0:5,:])

