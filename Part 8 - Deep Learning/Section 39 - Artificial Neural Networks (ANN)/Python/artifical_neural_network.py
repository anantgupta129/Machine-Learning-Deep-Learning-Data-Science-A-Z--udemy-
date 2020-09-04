# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 17:12:44 2020

@author: Dell
"""
                                # ARTIFICIAL NEURAL NETWORK

    # importing libraries    
import numpy as np
import pandas as pd
import tensorflow as tf

    # importing dataset
dataset= pd.read_csv('Churn_Modelling.csv')
x= dataset.iloc[:, 3:-1].values             # we don't need first 3 columns
y= dataset.iloc[:, -1].values

        # encoding categorical data
#label encoding (gender)
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
x[:, 2]= le.fit_transform(x[:, 2])

# oneHotEncoding (country)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct= ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x= np.array(ct.fit_transform(x))
"""one hot encoder encode like 001 100 010 
if there are 3 categorical variables """

        #Splitting into train and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state= 42)

        #feature scaling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train= sc.fit_transform(x_train)
x_test= sc.fit_transform(x_test)
'''in ANN we scale everything even categorical variables'''

                 #Building ANN
"""building artificial brain"""
    #initialize the ANN
ann= tf.keras.models.Sequential()
    #adding input layers and first hidden layer
ann.add(tf.keras.layers.Dense(units= 6, activation='relu'))  # relu= rectifier activation functions
    #adding input layers and first hidden layer
ann.add(tf.keras.layers.Dense(units= 6, activation='relu'))  #units are number of neurons
    #adding OUTPUT layer
ann.add(tf.keras.layers.Dense(units= 1, activation='sigmoid'))
"""with add function we can add a number of hidden layer anywhere"""

            #Training ANN
    #compile ANN
ann.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])
"""adam is mostly used in Stochastic Gradient Descent
categorical_crossentropy is used for categorical classification"""
    #training TRAIN set
ann.fit(x_train, y_train, batch_size=32, epochs=80)

"""Homework

Use our ANN model to predict if the customer with the following informations will leave the bank:

Geography: France

Credit Score: 600

Gender: Male

Age: 40 years old

Tenure: 3 years

Balance: $ 60000

Number of Products: 2

Does this customer have a credit card? Yes

Is this customer an Active Member: Yes

Estimated Salary: $ 50000

So, should we say goodbye to that customer?"""

print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) >0.5)
# print 0 or FALSE if less then 0.5(threshold)

        # PREDICTING test set result
y_pred= ann.predict(x_test)
y_pred= (y_pred > 0.6)
comp_pred_test= np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
print(comp_pred_test)

        # making CONFUSION MATRIX
from sklearn.metrics import confusion_matrix, accuracy_score
cm= confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred)*100,'%')



