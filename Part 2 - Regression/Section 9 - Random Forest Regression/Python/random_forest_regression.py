# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 19:40:55 2020

@author: Dell
"""
                    # Random Forest Regression
                    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dts = pd.read_csv('Position_Salaries.csv')
x = dts.iloc[:, 1:-1].values
y = dts.iloc[:, -1].values


'''Training the Random Forest Regression'''
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=30, random_state=0)
regressor.fit(x, y)

## Predicting a new result
regressor.predict([[6.5]])
    

##Visualising the Decision Tree Regression results (higher resolution)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


