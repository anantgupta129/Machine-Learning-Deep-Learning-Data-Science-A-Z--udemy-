                                  #DATA PREPROCESSING
        
        # IMPROTING LIBRARIES
import numpy as np                      #mathametical manipulation of arrays and matrix 
import matplotlib.pyplot as plt         #plotting
import pandas as pd                     #manage & import datasets 
        # IMPORTING DATASET
dataset = pd.read_csv("Data.csv")
x= dataset.iloc[:,:-1].values     #-1 mwans remove last column
y= dataset.iloc[:, -1].values
                # (pandas class )iloc allows us to take data by indexes
                # .values return values of column 
                # read_csv to read csv files (you can also put file name directlly if
                                             #it is in same folder with python file)
        
        # TAKING CARE OF MISING DATA
from sklearn.impute import SimpleImputer  # from is used to import a particular class or module
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
missingvalues = missingvalues.fit(x[:, 1:3])
x[:, 1:3] = missingvalues.transform(x[:, 1:3])
"""sklearn library tools for machine learning and statistical modeling
classification, regression, clustering and dimensionality reduction"""
                # SimpleImputer used to find mising vaules
      #class sklearn.impute.SimpleImputer(missing_values=nan, strategy='mean', fill_value=None, verbose=0, copy=True, add_indicator=False)
                # select missing vaule "nan" use strategy mean to replace we can also use median most_frequent whivh ever suits you
                # verbose 0 for column 1 for rows (mean)
                # fit will extract info of data imputer will spot mising value and find mean
                # transform will transform or replace the missing value    1:3 will select index 1 to 2

        # ENCODING CATEGORICAL DATA
  # Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
ct= ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x= np.array(ct.fit_transform(x), dtype=np.float)
"""one hot encoder is used when there are multiple categorical data 
like 3 or 4  countries"""
  
    # Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
"""label encoding is done when we need to apply labels like
0 and 1 for categorical data yes or no, male or female"""
                # ColumnTransformer Applies transformers to columns of an nparray or pandas DataFrame
                # LabelEncoder will give values like 0 1 2... for different variables, OneHotEncoder create dummy variables
                # 0 for selecting first column, passthrough to leave other column untouched 
                #OneHotEncoder to apply dummy np.array create arrays
"""fit_transform(x) method is called from the LabelEncoder() class, it transforms the categories 
strings into integers like France, Spain into 0 1 2...when called from the OneHotEncoder() class, 
it creates dummy variables different labels with binary values 0 and 1"""
                #   in dependent variables there is no use of creating dummy variables ML will know

        # SPILITTING DATASET into TRAIN and TEST SET
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42 )
                # 0.2 means 20% of data will test and remaining data 80% will be train 0.2 is a good number we can select any
                # 42 is ramdom state based on this it will 42 is a gernal number we go iwth any

        # FEATURE SCALING
from sklearn.preprocessing import StandardScaler    # hese we are using standardization featuring scaling
sc_x=StandardScaler()
x_train[:,3:5] = sc_x.fit_transform(x_train[:,3:5])
x_test[:,3:5] = sc_x.transform(x_test[:,3:5])       # test set are already fitted hence only transform
                # i have not applied scaling to dummy variables udemy has
"""we can apply standard scaler to dependent varibles if need here
dependent variables have categorical data so we don't"""




