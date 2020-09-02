# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 09:42:40 2020

@author: Dell
"""

                #SUPPORT VECTOR REGRESSION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dts = pd.read_csv('Position_Salaries.csv')
x = dts.iloc[:, 1:-1].values
y = dts.iloc[:, -1].values

y = np.reshape(y, (-1,1))

'''FEATURE SCALING'''
from sklearn.preprocessing import  StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)
# we create 2 different objects because both matrices
# has different features and diff mean and SD

'''TRAINING model by SVR'''
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x, y)

'''Prediciting Result'''
sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.4]])))


#'''VISUALISING SVR RESULTS'''
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x)), color = 'blue')
plt.title('Truth or Bluff (Support Vector Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#'''Visualising SVR results (for higher resolution and smoother curve)'''
x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))), color = 'blue')
plt.title('Truth or Bluff (Support Vector Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()



'''
SVR is good for mostly all kinds of dataset but if some points are really
scattered line position level 10 here then SVR won't catch it,
but since we wanted to predict the salary level at 6.5 that is close 
to all points then SVR gives better result without OVERfitting
'''

# CONCLUSION:
    #the choice of technique not only depends on data but also what and 
    #where we want to predict 
    
    
    
