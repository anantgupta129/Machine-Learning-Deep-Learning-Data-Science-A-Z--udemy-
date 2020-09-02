# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 10:28:40 2020

@author: Dell
"""
                #POLYNOMIAL LINEAR REGRESSION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dts = pd.read_csv('Position_Salaries.csv')
x = dts.iloc[:, 1:-1].values        #we dont need position as we already did like label encoding in LEVEL column
y = dts.iloc[:, -1].values

# we dont need to split in training and test set as dataset is small
y = np.reshape(y, (-1,1))

'''TRAINING model by LINEAR REGRESSION 1st'''

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

''' TRAINING model by POLYNOMIAL LINEAR REGRESSION '''
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2)
x_poly = poly.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)
# by PolynomialFeatures we actually transform matrix of fetures x to the new given degree
# and the fit it again using LinearRegression()

'''VISUALISING LINEAR REGRESSION RESULT '''
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

'''VISUALISING POLYNOMIAL LINEAR REGRESSION RESULT '''
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg2.predict(poly.fit_transform(x)), color = 'blue')
plt.title('Truth or Bluff (POLYNOMIAL Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

'''VISUALISING POLYNOMIAL LINEAR REGRESSION RESULT (with higher degree)'''
poly2 = PolynomialFeatures(degree = 4)
x_poly2 = poly2.fit_transform(x)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly2, y)
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg3.predict(poly2.fit_transform(x)), color = 'blue')
plt.title('Truth or Bluff (POLYNOMIAL Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# Predicting result with LINEAR REGRESSION
lin_reg.predict([[6.5]])
# Predicting result with PLOYNTMIAL LINEAR REGRESSION(degree= 2)
lin_reg2.predict(poly.fit_transform([[6.5]]))
# Predicting result with PLOYNTMIAL LINEAR REGRESSION(degree= 4)
lin_reg3.predict(poly2.fit_transform([[6.5]]))

'''Visualising Polynomial Regression results (for higher resolution and smoother curve)'''
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, lin_reg3.predict(poly2.fit_transform(x_grid)), color = 'blue')
plt.title('Truth or Bluff (POLYNOMIAL Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()




