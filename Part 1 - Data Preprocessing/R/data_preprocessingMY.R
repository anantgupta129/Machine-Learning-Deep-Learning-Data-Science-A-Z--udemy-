                # data preprocessing

      # importing data
dataset=read.csv("Data.csv")

      # taking care of mising data
# ifelse has three arguments first is condition sencond value to input if condition is true
# third is vaule to input is condition is false
dataset$Age=ifelse(is.na(dataset$Age),
                   ave(dataset$Age, FUN= function(x) mean(x, na.rm= TRUE)),
                   dataset$Age)
dataset$Salary=ifelse(is.na(dataset$Salary),
                      ave(dataset$Salary, FUN= function(x) mean(x, na.rm= TRUE)),
                      dataset$Salary)
#is.na will check if value is empty or not, ave() is used to calculate average


      # encoding categorical data
dataset$Country=factor(dataset$Country,
                       levels=c('France', 'Spain', 'Germany'),
                       labels=c(1, 2, 3))

dataset$Purchased=factor(dataset$Purchased,
                         levels=c('No', 'Yes'),
                         labels=c(0, 1))

      # splitting datasheet into train and test set
library(caTools)
set.seed(69)      # just like we selected random state in python
split = sample.split(dataset$Purchased, SplitRatio = 0.8)  
train_set = subset(dataset, split == TRUE) 
test_set = subset(dataset, split == FALSE) 
                  #sample.split returns true for TRUE and FALSE for test
                  # 0.8 means 80 data will be train

      # FEATUR SCALING
train_set[,2:3]= scale(train_set[,2:3])
test_set[,2:3]= scale(test_set[,2:3])




