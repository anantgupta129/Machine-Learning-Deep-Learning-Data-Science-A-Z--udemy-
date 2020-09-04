# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # APRiORi RULE LEARNiNg

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% [markdown]
# ## Data Preprocessing

# %%
dts = pd.read_csv("Market_Basket_Optimisation.csv", header = None)
transactions =[]
for i in range(0, len(dts)):
    transactions.append([str(dts.values[i, j]) for j in range(0, len(dts.columns))])

# %% [markdown]
# ## Training APRiORi on dataset

# %%
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)
"""
consider we want min 3 products per 3*7(weeks) 21/7501 =0.0027 ~ 0.03 i.e. min_support
min_confidence jut try different values till 0.8, min_length and max_length depends upon problem here we say that we saying buy 1 product and get 1 for free 
min_lift is just try different values 
"""

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
    lhs = [tuple(i[2][0][0])[0] for i in results]
    rhs = [tuple(i[2][0][1])[0] for i in results]
    support = [i[1] for i in results]
    confidence = [i[2][0][2] for i in results]
    lift = [i[2][0][3] for i in results]
    return list(zip(lhs, rhs, support, confidence, lift))
res_df = pd.DataFrame(inspect(results), columns= ["Left Hand Side", "Right Hand Side", "Support", "Confidence", "Lift"])


# %%
res_df

# %% [markdown]
# ### Displaying the results sorted by descending lifts

# %%
res_df.nlargest(columns= "Lift", n=10)


