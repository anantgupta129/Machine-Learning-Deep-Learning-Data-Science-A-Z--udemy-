# SOPPORT VECTOR Regression

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

#Feature Scaling
# Fitting SVR
library(e1071)
regressor = svm(formula = Salary ~. ,
                data = dataset, 
                type = 'eps-regression')
# type defines which kind of regression we are using live regression
# classification 
#in R SVM takes already scaled values

#PREDICTING the result
y_pred = predict(regressor, data.frame(Level = 6.5))

#Visualizing the result
library(ggplot2)
ggplot()+
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red')+
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
            colour = 'blue')+
  ggtitle('Truth or Bluff(SVR)')+
  xlab('Position')+
  ylab('Salary')







