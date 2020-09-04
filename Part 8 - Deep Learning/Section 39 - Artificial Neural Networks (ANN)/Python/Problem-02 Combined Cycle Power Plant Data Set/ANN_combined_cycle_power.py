# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 16:16:29 2020

@author: Dell
dataset link:- https://archive.ics.uci.edu/ml/datasets/Combined%2BCycle%2BPower%2BPlant
"""
                # ARTIFICIAL NEURAL NETWORK: Combined Cycle Power Plant Data Set
import numpy as np
import pandas as pd
import tensorflow as tf

dataset= pd.read_excel("Folds5x2_pp.xlsx")
x= dataset.iloc[:, :-1].values
y= dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state= 42)

        #BULIDING ANN
    #initialize ANN
ann= tf.keras.models.Sequential()
    #adding input layer and 1st hidden layer
ann.add(tf.keras.layers.Dense(units= 7, activation='relu'))
    #adding 2nd hidden layer
ann.add(tf.keras.layers.Dense(units= 6, activation='relu'))
    #output layer
ann.add(tf.keras.layers.Dense(units= 1))
''' sigmoid when 2 prediction like 0 and 1soft max when more then 1 dependent variable
are used in classificatin
in regression no activation function'''
        #TRAINING ANN
    #Compile ANN
ann.compile(optimizer= 'adam', loss='mean_squared_error')
    #training
ann.fit(x_train, y_train, batch_size=32, epochs=100)

        #PREDICTING TEST SET RESULT
y_pred= ann.predict(x_test)
np.set_printoptions(precision=3)
compare_res= np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
print(compare_res)






