# RANDOM FOREST REgression

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]


# Fitting Decision Tree REgression
library(randomForest)
set.seed(1234)
regressor = randomForest(x = dataset[1],
                         y = dataset$Salary,
                         ntree = 500)
#ntree sets number of tree

####PREDICTING the result
y_pred = predict(regressor, data.frame(Level = 6.5))

#Visualizing the result
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot()+
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red')+
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue')+
  ggtitle('Truth or Bluff(Random ForestRegression)')+
  xlab('Position')+
  ylab('Salary')




