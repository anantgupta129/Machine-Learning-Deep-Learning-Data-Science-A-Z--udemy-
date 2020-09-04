    # HIERARCHICAL CLUSTERiNG

dataset = read.csv("Mall_Customers.csv")
x = dataset[4:5]

# building DENDOGRAM
dendogram = hclust(dist(x, method = 'euclidean'), method = 'ward.D')
plot(dendogram, 
     main = paste('DEndoGram'),
     xlab = 'Customers', ylab = 'euclidean distance')

# Fitting Hierarchical CLuster to data 
hc = hclust(dist(x, method = 'euclidean'), method = 'ward.D')
y_hc = cutree(hc, 5)

# Visualizing Clusters
library(cluster)
clusplot(x, y_hc,
         lines = 0,
         shade = TRUE, color = TRUE, 
         labels = 2, plotchar = FALSE,
         span = TRUE,
         main = paste('Cluster of Customers'),
         xlab = 'Annual INcome', ylab = 'Spending SCore')

