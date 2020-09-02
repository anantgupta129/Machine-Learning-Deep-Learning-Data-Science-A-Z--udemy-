# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 17:51:53 2020

@author: Dell
"""
                                # SIMPLE LINEAR REGRESSION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

 
dataset = pd.read_csv("D:\Course\Machine learning\Machine Learning A-Z\Part 2 - Regression\Section 4 - Simple Linear Regression\Salary_Data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=1/3, random_state=0)


        #fitting simple regression to training set
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)

y_pred= reg.predict(x_test)        #predicting test results

        #plotting train results
plt.scatter(x_train, y_train, color= 'red')
plt.plot(x_train, reg.predict(x_train), color= 'blue')  
plt.title('Experience vs salary')
plt.xlabel('Experience in Years')
plt.ylabel('Salary')

plt.scatter(x_test, y_test, color= 'orange')   #plotting test results

plt.show()

""" here didn't apply plt.plot(x_train, reg.predict(x_train), color= 'blue')  as test as 
we are training there and then test the results so there is no need of reg.predict(x_train)
as it will be like training again """