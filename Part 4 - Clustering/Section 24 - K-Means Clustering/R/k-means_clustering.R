      # K-MEANS CLUSTERING

dataset = read.csv('Mall_Customers.csv')
x = dataset[,4:5]

# USing the Elbow MEthod To FiND optimal number Of CluSTERs
set.seed(6)
wcss = vector()
for (i in 1:10) wcss[i] = sum(kmeans(x, i)$withinss)
plot(1:10, wcss, type = "b", main = paste('Clusters of clients'), 
     xlab = "Number of CLusters", ylab = 'WCSS')

# Applying K-Means to dataset
set.seed(29)
kmeans = kmeans(x, 5, iter.max = 300, nstart = 10)

# Visualizing CLuster
library(cluster)
clusplot(x, kmeans$cluster,
         lines = 0,
         shade = TRUE, color = TRUE, labels = 2,
         plotchar = FALSE, span = TRUE, 
         main = paste("Clusters of clients"),
         xlab = 'Anual Income',
         ylab = 'Spending Score')


