      # APRIORI ASSOCIATION RULE LEARNING
dataset = read.csv('Market_BASket_Optimisation.csv', header = FALSE)
library(arules)
dataset = read.transactions('Market_BASket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN=20)

# training apriori model on dataset
rules = apriori(data = dataset, parameter = list(support = 0.003, confidence = 0.2))

# visualasing
inspect(sort(rules, by = 'lift')[1:10])
