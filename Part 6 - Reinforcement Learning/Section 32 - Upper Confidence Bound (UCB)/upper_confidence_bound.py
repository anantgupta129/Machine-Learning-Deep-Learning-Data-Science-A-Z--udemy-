# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # UPPER CONFiDENCE BOUNd (UCB)

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


# %%
dts = pd.read_csv("Ads_CTR_Optimisation.csv")

# %% [markdown]
# ## Implementing UCB

# %%
N = len(dts)                        # number of users
d = len(dts.columns)                # no of ads
ads_selected = []
numbers_of_selections = [0] *d
sum_of_rewards = [0] *d
total_reward = 0

for n in range(N):
    ad = 0
    max_ucb = 0
    for i in range(d):
        if (numbers_of_selections[i] != 0) :
            average_reward = sum_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 *math.log(n + 1) / numbers_of_selections[i])
            ucb = average_reward + delta_i
        else:
            ucb = 1e400
        if (ucb > max_ucb):
            max_ucb = ucb
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1 
    reward = dts.values[n, ad]
    sum_of_rewards[ad] += reward
    total_reward += reward


# %%
print(numbers_of_selections)
print(sum_of_rewards)
print(total_reward)

# %% [markdown]
# ## Visualising Results

# %%
plt.hist(ads_selected)
plt.title("Histogram of Ads Selection")
plt.xlabel('Ads')
plt.ylabel('NUmber of times each Ad was selected')
plt.show()

