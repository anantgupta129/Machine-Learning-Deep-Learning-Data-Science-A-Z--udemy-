# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # ECLAT ASSOCiATiOn RULE LEARNiNg

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# %%
dts = pd.read_csv("Market_Basket_Optimisation.csv", header = None)
transactions =[]
for i in range(0, len(dts)):
    transactions.append([str(dts.values[i, j]) for j in range(0, len(dts.columns))])

# %% [markdown]
# ## Training ECLAT Model on dataset

# %%
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

# %% [markdown]
# ## ViSualising REsults
# %% [markdown]
# ### Displaying the first results coming directly from the output of the apriori function

# %%
results = list(rules)

# %% [markdown]
# ### Putting the results well organised into a Pandas DataFrame & DiSPlaYiNG

# %%
def inspect(results):
    p1 = [tuple(i[2][0][0])[0] for i in results]
    p2 = [tuple(i[2][0][1])[0] for i in results]
    support = [i[1] for i in results]
    return list(zip(p1, p2, support))
res_df = pd.DataFrame(inspect(results), columns= ["Product 1", "Product 2", "Support"])

# %% [markdown]
# ### Displaying the results sorted by descending Support

# %%
res_df.nlargest(columns= "Support", n=10)


