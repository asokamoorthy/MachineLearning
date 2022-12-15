# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 06:13:41 2018

@author: Asokamoorthy K
"""
"""
Data Set Information:

The file "sonar.mines" contains 111 patterns obtained by bouncing sonar signals off a metal cylinder at various angles and under 
various conditions. The file "sonar.rocks" contains 97 patterns obtained from rocks under similar conditions. 
The transmitted sonar signal is a frequency-modulated chirp, rising in frequency. The data set contains signals 
obtained from a variety of different aspect angles, spanning 90 degrees for the cylinder and 180 degrees for the rock. 

Each pattern is a set of 60 numbers in the range 0.0 to 1.0. Each number represents the energy within a particular
 frequency band, integrated over a certain period of time. The integration aperture for higher frequencies occur 
 later in time, since these frequencies are transmitted later during the chirp. 

The label associated with each record contains the letter "R" if the object is a rock and "M" if it is a mine (metal cylinder).
 The numbers in the labels are in increasing order of aspect angle, but they do not encode the angle directly.
"""
# Load libraries 
import numpy 
from matplotlib import pyplot 
from pandas import read_csv 
from pandas import set_option 
from pandas.plotting import scatter_matrix 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.ensemble import RandomForestClassifier  
from sklearn.ensemble import ExtraTreesClassifier


url = 'sonar.all-data.csv' 
dataset = read_csv(url, header=None)

# shape 
print(dataset.shape)

# types 
set_option('display.max_rows', 500) 
print(dataset.dtypes)

# head 
set_option('display.width', 100) 
print(dataset.head(10))

# descriptions, change precision to 3 places 
set_option('precision', 3) 
print(dataset.describe())


# class distribution 
print(dataset.groupby(60).size())


# histograms 
#dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
#pyplot.show()

dataset[25].hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
pyplot.show()


# correlation matrix 
fig = pyplot.figure() 
ax = fig.add_subplot(111) 
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none') 
fig.colorbar(cax) 
pyplot.show()

# Split-out validation dataset 
array = dataset.values 
X = array[:,0:60].astype(float) 
Y = array[:,60] 
validation_size = 0.20 
seed = 7 
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Test options and evaluation metric 
num_folds = 10 
seed = 7 
scoring = 'accuracy'

# Spot-Check Algorithms 
models = [] 
models.append(('LR', LogisticRegression())) 
models.append(('LDA', LinearDiscriminantAnalysis())) 
models.append(('KNN', KNeighborsClassifier())) 
models.append(('CART', DecisionTreeClassifier())) 
models.append(('NB', GaussianNB())) 
models.append(('SVM', SVC()))

results = [] 
names = [] 
for name, model in models: 
    kfold = KFold(n_splits=num_folds, random_state=seed) 
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring) 
    results.append(cv_results) 
    names.append(name) 
    msg = "%s: %f (%f)" % (names, cv_results.mean(), cv_results.std()) 
    print(msg)
    #print(cv_results.mean())
    #print(cv_results.std())

"""
The results suggest That both Logistic Regression and k-Nearest Neighbors may be worth further study.
"""

# Compare Algorithms 
fig = pyplot.figure() 
fig.suptitle('Algorithm Comparison') 
ax = fig.add_subplot(111) 
pyplot.boxplot(results) 
ax.set_xticklabels(names) 
pyplot.show()

"""
The results show a tight distribution for KNN which is encouraging, suggesting low variance.
 The poor results for SVM are surprising.
"""

# Standardize the dataset 
pipelines = [] 
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LogisticRegression())]))) 

pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))

pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())]))) 
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())]))) 
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())]))) 
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC())]))) 


results = [] 
names = [] 
for name, model in pipelines: 
    kfold = KFold(n_splits=num_folds, random_state=seed) 
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring) 
    results.append(cv_results) 
    names.append(name) 
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
    print(msg)
"""
 We can see that KNN is still doing well, even better than before. We can also see that 
 the standardization of the data has lifted the skill of SVM to be the most accurate algorithm tested so far
 """
 
 # Compare Algorithms 
fig = pyplot.figure() 
fig.suptitle('Scaled Algorithm Comparison') 
ax = fig.add_subplot(111) 
pyplot.boxplot(results) 
ax.set_xticklabels(names) 
pyplot.show() 

# Tune scaled KNN 
scaler = StandardScaler().fit(X_train) 
rescaledX = scaler.transform(X_train) 

neighbors = [1,3,5,7,9,11,13,15,17,19,21] 
param_grid = dict(n_neighbors=neighbors) 

model = KNeighborsClassifier() 
kfold = KFold(n_splits=num_folds, random_state=seed) 
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold) 
grid_result = grid.fit(rescaledX, Y_train) 
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score'] 
stds = grid_result.cv_results_['std_test_score'] 
params = grid_result.cv_results_['params'] 
for mean, stdev, param in zip(means, stds, params): 
    print("%f (%f) with: %r" % (mean, stdev, param))
    
"""
Another way that we can improve the performance of 
algorithms on this problem is by using ensemble methods. 

"""


# ensembles 
ensembles = [] 
ensembles.append(('AB', AdaBoostClassifier())) 
ensembles.append(('GBM', GradientBoostingClassifier())) 
ensembles.append(('RF', RandomForestClassifier())) 
ensembles.append(('ET', ExtraTreesClassifier())) 

results = [] 
names = [] 
for name, model in ensembles: 
    kfold = KFold(n_splits=num_folds, random_state=seed) 
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring) 
    results.append(cv_results) 
    names.append(name) 
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
    print(msg)
    

# Compare Algorithms 
fig = pyplot.figure() 
fig.suptitle('Ensemble Algorithm Comparison') 
ax = fig.add_subplot(111) 
pyplot.boxplot(results) 
ax.set_xticklabels(names) 
pyplot.show()

"""
The results suggest GBM may be worthy of further study, with a strong mean and a spread that skews up towards high 90s (%) in accuracy

The SVM showed the most promise as a low complexity and stable model for this problem. 
In this section we will ﬁnalize the model by training it on 
the entire training dataset and make predictions for the hold-out validation
 dataset to conﬁrm our ﬁndings. A part of the ﬁndings was that SVM performs better 
 when the dataset is standardized so that all attributes have a mean value of zero 
 and a standard deviation of one. We can calculate this from the entire training dataset and 
 apply the same transform to the input attributes from the validation dataset.

"""
# prepare the model 
scaler = StandardScaler().fit(X_train) 
rescaledX = scaler.transform(X_train) 
model = SVC(C=1.5) 
model.fit(rescaledX, Y_train) 
# estimate accuracy on validation dataset
rescaledValidationX = scaler.transform(X_validation) 

predictions = model.predict(rescaledValidationX) 
print(accuracy_score(Y_validation, predictions)) 
print(confusion_matrix(Y_validation, predictions)) 
print(classification_report(Y_validation, predictions))

"""
We can see that we achieve an accuracy of nearly 86% on the held-out validation dataset. 
A score that matches closely to our expectations estimated above during the tuning of SVM
"""







