      # ECLAT
dataset = read.csv('Market_BASket_Optimisation.csv', header = FALSE)
library(arules)
dataset = read.transactions('Market_BASket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN=20)

# training ECLAT model on dataset
rules = eclat(data = dataset, parameter = list(support = 0.004, minlen =2))

# visualizing
inspect(sort(rules, by = 'support')[1:10])
