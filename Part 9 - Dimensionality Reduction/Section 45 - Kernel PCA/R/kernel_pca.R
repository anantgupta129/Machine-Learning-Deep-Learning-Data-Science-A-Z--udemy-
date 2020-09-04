#Kernel PCA

dts = read.csv('Social_Network_Ads.csv')
dts = dts[, 3:5]


#splitting dataset into Train set and test set
library(caTools)
set.seed(69)
split = sample.split(dts$Purchased, SplitRatio=0.75)
train_set = subset(dts, split==TRUE)
test_set = subset(dts, split==FALSE)

#FEATURE scaling
train_set[, 1:2] = scale(train_set[, 1:2])
test_set[, 1:2] = scale(test_set[, 1:2])

# Kernel PCA
library(kernlab)
kpca = kpca(~., data = train_set[-3], kernel = 'rbfdot', features = 2)
training_set_pca = as.data.frame(predict(kpca, train_set))
training_set_pca$Purchased = train_set$Purchased
test_set_pca = as.data.frame(predict(kpca, test_set))
test_set_pca$Purchased = test_set$Purchased

#FITTING logistic REGRESSION
classifier = glm(formula = Purchased ~ .,
                 family = binomial,
                 data = training_set_pca)

#PREDICTING RESULTS of TEST set
prob_pred = predict(classifier, type = 'response', newdata = test_set_pca[-3])
y_pred = ifelse(prob_pred > 0.5, 
                1, 0 )

#making CONFUSION MATRIX
cm = table(test_set[,3], y_pred)

#VISUALISING train RESULTS
library(ElemStatLearn)
set = training_set_pca
x1 = seq(min(set[,1]) -1, max(set[,1]) +1, by=0.01)
x2 = seq(min(set[,2]) -1, max(set[,2]) +1, by=0.01)
grid_set = expand.grid(x1, x2)
colnames(grid_set) = c('V1', 'V2')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 
                1, 0 )

plot(set[,-3],
     main = 'Logistic Regression(Train Set)',
     xlab = 'V1', ylab = 'V2',
     xlim = range(x1), ylim = range(x2))
contour(x1, x2, matrix(as.numeric(y_grid), length(x1), length(x2)), add = TRUE)
points(grid_set, pch ='.', col= ifelse(y_grid ==1, 'springgreen3', 'tomato'))
points(set, pch = 21, col= ifelse(set[, 3] ==1, 'green4', 'red3'))

#VISUALISING TEST RESULTS
library(ElemStatLearn)
set = test_set_pca  
x1 = seq(min(set[,1]) -1, max(set[,1]) +1, by=0.01)
x2 = seq(min(set[,2]) -1, max(set[,2]) +1, by=0.01)
grid_set = expand.grid(x1, x2)
colnames(grid_set) = c('V1', 'V2')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 
                1, 0 )

plot(set[,-3],
     main = 'Logistic Regression(Test Set)',
     xlab = 'V1', ylab = 'V2',
     xlim = range(x1), ylim = range(x2))
contour(x1, x2, matrix(as.numeric(y_grid), length(x1), length(x2)), add = TRUE)
points(grid_set, pch ='.', col= ifelse(y_grid ==1, 'springgreen3', 'tomato'))
points(set, pch = 21, col= ifelse(set[, 3] ==1, 'green4', 'red3'))


