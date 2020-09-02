                # SIMPLE LINEAR REGRESSION
dataset= read.csv('Salary_data.csv')

library(caTools)
set.seed(69)
split = sample.split(dataset$Salary, SplitRatio= 2/3)
train_set = subset( dataset, split== TRUE)
test_set= subset(dataset, split== FALSE)

  # Fitting Simple Linear Regression to the Training set
regressor = lm(formula = Salary ~ YearsExperience,
               train_set)
# apply regressor to test set
y_pred= predict(regressor, newdata= test_set)

#plotting results
library(ggplot2)
ggplot()+
  geom_point(aes(x= train_set$YearsExperience, y= train_set$Salary),
             color= 'red')+
  geom_line(aes(x= train_set$YearsExperience, y= predict(regressor, newdata= train_set) ),
            color= 'blue')+
  geom_smooth(method='lm')+
  ggtitle('Salary vs Years of Experience')+
  xlab('Years of Experience')+
  ylab('Salary')+
  geom_point(aes(x= test_set$YearsExperience, y= test_set$Salary),
             color = 'orange')


# aes is aesthetics function used to give coordinate in ggplot 
  

