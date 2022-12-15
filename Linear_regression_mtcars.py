# -*- coding: utf-8 -*-
"""
Created on Wed May 30 20:59:02 2018

@author: Asokamoorthy K
"""

# to get built in data set

# pip install pydataset


"""
Assignment
---------
Assignment

1. variable type
2. libraries
3. what is simple linear regression

Template
=======
declare libraries

read dataset

split x,y

fit the model - Train

prediction

error
------------------

Description
The data was extracted from the 1974 Motor Trend US magazine, 
and comprises fuel consumption and 10 aspects of automobile design and performance for 32 automobiles
 (1973–74 models).

Usage
mtcars
Format
A data frame with 32 observations on 11 (numeric) variables.

[, 1]	mpg	Miles/(US) gallon
[, 2]	cyl	Number of cylinders
[, 3]	disp	Displacement (cu.in.)
[, 4]	hp	Gross horsepower
[, 5]	drat	Rear axle ratio
[, 6]	wt	Weight (1000 lbs)
[, 7]	qsec	1/4 mile time
[, 8]	vs	Engine (0 = V-shaped, 1 = straight)
[, 9]	am	Transmission (0 = automatic, 1 = manual)
[,10]	gear	Number of forward gears
[,11]	carb	Number of carburetors

------------------------
Linear Regression Basics
Linear regression is a predictive modeling technique for predicting 
a numeric response variable based on one or more explanatory variables. 
The term "regression" in predictive modeling generally refers to any modeling task 
that involves predicting a real number (as opposed classification, 
which involves predicting a category or class.). 
The term "linear" in the name linear regression refers to the fact that
 the method models data with linear combination of the explanatory variables. 
 A linear combination is an expression where one or more variables are scaled
 by a constant factor and added together. In the case of linear regression
 with a single explanatory variable, the linear combination used in
 linear regression can be expressed as:

response=intercept+constant∗explanatory
y = mx1+c
Y= B0 + B1weight

The right side if the equation defines a line with a certain 
y-intercept and slope times the explanatory variable.
 In other words, linear regression in its most basic 
 form fits a straight line to the response variable.
 The model is designed to fit a line that minimizes 
 the squared differences (also called errors or residuals.).
 We won't go into all the math behind how the model actually
 minimizes the squared errors, but the end result is a line
 intended to give the "best fit" to the data. Since linear 
 regression fits data with a line, it is most effective in cases
 where the response and explanatory variable have a linear 
 relationship.

 y = mx+c
 
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model # sci-kit learn
from pydataset import data

m = data('mtcars')

data = pd.DataFrame(m)
data

data('mpg')

data.shape
data.isnull().count()

data.plot(kind="scatter",  x="wt",  y="mpg",  figsize=(9,9),  color="black")

# Train the model using the mtcars data
X = pd.DataFrame(data["wt"])
y = data["mpg"]

# Initialize model
regression_model = linear_model.LinearRegression()

# Trained
regression_model.fit(X,y)

Y_predict = regression_model.predict(X)
Y_predict

data2 = pd.read_csv("book2.csv")

Y_predict = regression_model.predict(data2)
Y_predict




# Check trained model y-intercept
print(regression_model.intercept_) 

# Check trained model coefficients
print(regression_model.coef_)

yp = 37.2851 + (-5.3444 * 1.5) + (-5.34 * 4) + (-9.3 * 120)
y = B0+ b1X1 + B2X2 +b3X3
 
yp
"""
The output above shows the model intercept and coefficients used to create
 the best fit line. In this case the y-intercept term is 
 set to 37.2851 and the coefficient for the weight variable is -5.3445. 
 In other words, the model fit the line mpg = 37.2851 - 5.3445*wt.
"""

regression_model.score(X = pd.DataFrame(data["wt"]), 
                       y = data["mpg"])

Y_predict = regression_model.predict(X = pd.DataFrame(data["wt"]))
Y_predict

# Actual - prediction = residuals
residuals = Y_predict - data["mpg"]

residuals[0]


from sklearn.linear_model import Lasso
from sklearn.cross_validation import train_test_split


lreg = Lasso(alpha=0.3, normalize=True)

data.head(5)
X = data.loc[:, data.columns != 'mpg']
y = data.loc[:, data.columns == 'mpg']


x_train, x_cv, y_train, y_cv = train_test_split(X, y,test_size =0.3)

lreg.fit(x_train,y_train)




pred_cv = lreg.predict(x_cv)

mse = np.mean((pred_cv - y_cv)**2)

1346205.82

lassoReg.score(x_cv,y_cv)


predictors = x_train.columns

coef = Series(lreg.coef_,predictors).sort_values()

coef.plot(kind='bar', title='Modal Coefficients')


lreg.fit(x_train,y_train)

# predicting on cv



# calculating mse
"""
Linear relationship
Multivariate normality
No or little multicollinearity
No auto-correlation
Homoscedasticity
Home | Academic Solutions | Directory of Statistical Analyses | Regression Analysis | Assumptions of Linear Regression
Assumptions of Linear Regression
Linear regression is an analysis that assesses whether one or more predictor variables explain the dependent (criterion) variable.  The regression has five key assumptions:

Linear relationship
Multivariate normality
No or little multicollinearity
No auto-correlation
Homoscedasticity
 

A note about sample size.  In Linear regression the sample size rule of thumb is that the regression analysis requires at least 20 cases per independent variable in the analysis.

In the free software below, its really easy to conduct a regression and most of the assumptions are preloaded and interpreted for you.


Name
Email
First, linear regression needs the relationship between the independent and dependent variables to be linear.  It is also important to check for outliers since linear regression is sensitive to outlier effects.  The linearity assumption can best be tested with scatter plots, the following two examples depict two cases, where no and little linearity is present.



Secondly, the linear regression analysis requires all variables to be multivariate normal.  This assumption can best be checked with a histogram or a Q-Q-Plot.  Normality can be checked with a goodness of fit test, e.g., the Kolmogorov-Smirnov test.  When the data is not normally distributed a non-linear transformation (e.g., log-transformation) might fix this issue.



Thirdly, linear regression assumes that there is little or no multicollinearity in the data.  Multicollinearity occurs when the independent variables are too highly correlated with each other.

Multicollinearity may be tested with three central criteria:

1) Correlation matrix – when computing the matrix of Pearson’s Bivariate Correlation among all independent variables the correlation coefficients need to be smaller than 1.

2) Tolerance – the tolerance measures the influence of one independent variable on all other independent variables; the tolerance is calculated with an initial linear regression analysis.  Tolerance is defined as T = 1 – R² for these first step regression analysis.  With T < 0.1 there might be multicollinearity in the data and with T < 0.01 there certainly is.

3) Variance Inflation Factor (VIF) – the variance inflation factor of the linear regression is defined as VIF = 1/T. With VIF > 10 there is an indication that multicollinearity may be present; with VIF > 100 there is certainly multicollinearity among the variables.

If multicollinearity is found in the data, centering the data (that is deducting the mean of the variable from each score) might help to solve the problem.  However, the simplest way to address the problem is to remove independent variables with high VIF values.

Fourth, linear regression analysis requires that there is little or no autocorrelation in the data.  Autocorrelation occurs when the residuals are not independent from each other.  For instance, this typically occurs in stock prices, where the price is not independent from the previous price.

4) Condition Index – the condition index is calculated using a factor analysis on the independent variables.  Values of 10-30 indicate a mediocre multicollinearity in the linear regression variables, values > 30 indicate strong multicollinearity.

If multicollinearity is found in the data centering the data, that is deducting the mean score might help to solve the problem.  Other alternatives to tackle the problems is conducting a factor analysis and rotating the factors to insure independence of the factors in the linear regression analysis.

Fourthly, linear regression analysis requires that there is little or no autocorrelation in the data.  Autocorrelation occurs when the residuals are not independent from each other.  In other words when the value of y(x+1) is not independent from the value of y(x).



While a scatterplot allows you to check for autocorrelations, you can test the linear regression model for autocorrelation with the Durbin-Watson test.  Durbin-Watson’s d tests the null hypothesis that the residuals are not linearly auto-correlated.  While d can assume values between 0 and 4, values around 2 indicate no autocorrelation.  As a rule of thumb values of 1.5 < d < 2.5 show that there is no auto-correlation in the data. However, the Durbin-Watson test only analyses linear autocorrelation and only between direct neighbors, which are first order effects.

The last assumption of the linear regression analysis is homoscedasticity.  The scatter plot is good way to check whether the data are homoscedastic (meaning the residuals are equal across the regression line).  The following scatter plots show examples of data that are not homoscedastic (i.e., heteroscedastic):



The Goldfeld-Quandt Test can also be used to test for heteroscedasticity.  The test splits the data into two groups and tests to see if the variances of the residuals are similar across the groups.  If homoscedasticity is present, a non-linear correction might fix the problem.
"""