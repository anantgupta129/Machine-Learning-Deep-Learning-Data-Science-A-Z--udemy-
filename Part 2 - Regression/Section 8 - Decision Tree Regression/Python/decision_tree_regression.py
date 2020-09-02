# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 12:20:47 2020

@author: Dell
"""
                    # Decision Tree Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dts = pd.read_csv('Position_Salaries.csv')
x = dts.iloc[:, 1:-1].values
y = dts.iloc[:, -1].values

'''Training the Decision Tree Regression'''
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x, y)

## Predicting a new result
regressor.predict([[6.5]])


##Visualising the Decision Tree Regression results (higher resolution)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.title('Truth or Bluff (Support Vector Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


'''
Decission Tree regression is better when there are several
features'''
'''
There is no need of feature scaling as there is no formulae used 
prediction is just based on splits/distribution'''

