#POLYNOMIAL LINEAR REGRESSION

dts = read.csv('Position_Salaries.csv')
dts = dts[2:3]

lin_reg = lm(formula = Salary ~. ,
             data = dts)

# Fitting Polynomial Regression
dts$Level2 = dts$Level^2
dts$Level3 = dts$Level^3
dts$Level4 = dts$Level^4
poly_reg = lm(formula = Salary ~. ,
              data = dts)

summary(poly_reg)

#Visualizing Linear REgression
library(ggplot2)
ggplot()+
  geom_point(aes(x= dts$Level, y= dts$Salary),
             colour = 'red')+
  geom_line(aes(x= dts$Level, y= predict(lin_reg, newdata = dts)),
            colour = 'blue')+
  ggtitle('Truth or Bluff (Linear Regression)')+
  xlab('Level')+
  ylab('Salary')


#Visualizing POLYNOMIAL REgression
library(ggplot2)
ggplot()+
  geom_point(aes(x= dts$Level, y= dts$Salary),
             colour = 'red')+
  geom_line(aes(x= dts$Level, y= predict(poly_reg, newdata = dts)),
            colour = 'blue')+
  ggtitle('Truth or Bluff (Polynomial Regression)')+
  xlab('Level')+
  ylab('Salary')

# Predicting Result with Linear Regression
predict(lin_reg, data.frame(Level = 6.5))

# Predicting Result with Polynomial Regression
predict(poly_reg, data.frame(Level = 6.5,
                             Level2 =6.5^2,
                             Level3 =6.5^3,
                             Level4 =6.5^4))



