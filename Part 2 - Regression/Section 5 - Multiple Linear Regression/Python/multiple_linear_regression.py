                # Multiple Linear Regression

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

 
dataset = pd.read_csv("50_Startups.csv")
x= dataset.iloc[:, :-1].values
y= dataset.iloc[:, -1].values

# Encoding CATEGORICAL data independent variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct= ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder= 'passthrough')
x = np.array(ct.fit_transform(x) , dtype = np.float)

"""
oneHotEncoder take care of DUMMUY trap on its own

--there is no need of feature caling in Multiple regression because cofficients
 will compensate it
"""

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
reg= LinearRegression()
reg.fit(x_train, y_train)

# Predicting the Test set results
y_pred = reg.predict(x_test)
np.set_printoptions(precision= 3)   # decimals to 3 
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

            #concatenate is used to join different arrays
        
'''QUESTION
Making a single prediction (for example the profit of a startup with R&D Spend = 160000,
 Administration Spend = 130000, Marketing Spend = 300000 and State = 'California')'''        
        
print(reg.predict([[1, 0, 0, 160000, 130000, 300000]]))


'''Important note 1: Notice that the values of the features were all input in a double pair of square
brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. 
And putting our values into a double pair of square brackets makes the input exactly a 2D array. Simply put:

1,0,0,160000,130000,300000→scalars

[1,0,0,160000,130000,300000]→1D array

[[1,0,0,160000,130000,300000]]→2D array

Important note 2: Notice also that the "California" state was not input as a string in the last 
column but as "1, 0, 0" in the first three columns. That's because of course the predict method expects the 
one-hot-encoded values of the state, and as we see in the second row of the matrix of features X, "California" 
was encoded as "1, 0, 0". And be careful to include these values in the first three columns, not the last three ones, 
because the dummy variables are always created in the first columns'''

print(reg.coef_)
print(reg.intercept_)

'''Therefore, the equation of our multiple linear regression model is:

Profit=86.6×Dummy State 1−873×Dummy State 2+786×Dummy State 3−0.773×R&D Spend+0.0329×Administration+0.0366×Marketing Spend+42467.53

Important Note: To get these coefficients we called the "coef_" and "intercept_" attributes from our regressor 
object. Attributes in Python are different than methods and usually return a simple value or an array of values.'''

#buliding optimal model using BACKWARD ELIMINATION
import statsmodels.api as sm
x = np.append(arr = np.ones((50, 1)).astype(int), values=x, axis=1) 
x_opt =x[:,[0, 1, 2, 3, 4, 5, 6]]
regressor_OLS = sm.OLS(endog = y, exog= x_opt).fit()
regressor_OLS.summary()
x_opt =x[:,[0, 1, 2, 3, 4, 6]]  #removing 5 column as it has greter p value
regressor_OLS = sm.OLS(endog = y, exog= x_opt).fit()
regressor_OLS.summary()
x_opt =x[:,[0, 1, 2, 3, 4]]  #removing 6 column as it has greter p value
regressor_OLS = sm.OLS(endog = y, exog= x_opt).fit()
regressor_OLS.summary()

# automatic backward elemination
import statsmodels.api as sm

def backwardElimination(a, sl = 0.05):
    for i in range(0, len(a[0])):
        regressor_OLS = sm.OLS(y, a).fit()
        maxPvalue = max(regressor_OLS.pvalues).astype(float)
        if maxPvalue> sl:
            for j in range(0, len(a[0]-i)):
                if (regressor_OLS.pvalues[j].astype(float) == maxPvalue):
                    a = np.delete(a, j, 1)
    regressor_OLS.summary()
    return a

sl = 0.05
x_opt =x[:,[0, 1, 2, 3, 4, 5, 6]]
x_modelled = backwardElimination(x_opt, sl)

y_pred2 = regressor_OLS.predict(x_modelled)

