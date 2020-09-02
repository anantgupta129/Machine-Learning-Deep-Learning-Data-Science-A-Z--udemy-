
dataset= read.csv('50_Startups.csv')

#categorical data
dataset$State= factor(dataset$State,
                      levels = c('New York', 'California', 'Florida'),
                      labels = c(1, 2, 3))

#Splitting dataset
library(caTools)
set.seed(69)
split= sample.split(dataset$Profit, SplitRatio = 0.8)
train_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Fitting model to training set
regressor = lm(formula = Profit ~ .,
               data = train_set)
# . represents all independent variables we can also write like a+b+c
summary(regressor)

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)

#BACKWARD ELIMINATION

regressor = lm(formula = Profit ~ R.D.Spend+ Administration+ Marketing.Spend+ State,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend+ Administration+ Marketing.Spend,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend+ Marketing.Spend,
               data = dataset)
summary(regressor)
#removing independent variables one by one which has p-value higher then 0.05

y_pred2 = predict(regressor, newdata = test_set)

#Automatic BACKWARD ELIMINATION
backwardElimination <- function(x, sl) {
  len = length(x)
  for (i in c(1:len)) {
    regressor = lm(formula = Profit ~., data = x)
    maxVar = max(coef(summary(regressor))[c(2:len), "Pr(>|t|)"])
    if (maxVar >sl){
      j = which(coef(summary(regressor))[c(2:len),"Pr(>|t|)"]==maxVar)
      x = x[, -j]
    }
    len = len-1
  }
  return(summary(regressor))
}

sl = 0.05
dts = dataset
dts_modeled= backwardElimination(dts, sl)

y_pred3 = predict(regressor, newdata = test_set)


