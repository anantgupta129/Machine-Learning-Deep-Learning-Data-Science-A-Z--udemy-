# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # HiERARcHiCAl ClUSTeRiNG

# %%
import pandas as pd
import matplotlib.pylab as plt
import numpy as np


# %%
dts = pd.read_csv("Mall_Customers.csv")
x = dts.iloc[:, [3, 4]].values

# %% [markdown]
# ## Using DeNDoGRAm to find optimal number of CLUSteRs

# %%
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(x, method= 'ward'))
plt.title('DeNdoGram')
plt.xlabel('Cutomers')
plt.ylabel('Euclidean Distances')
plt.show()

# %% [markdown]
# ## training hierarchical model on dataset

# %%
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters= 5, linkage= 'ward', affinity= 'euclidean')
y_hc = hc.fit_predict(x)


# %%
y_hc

# %% [markdown]
# ## Visualizing Cluster

# %%
color_list = ['red', 'blue', 'green', 'cyan', 'magenta']
cluster_label = ['CLuster 1', 'CLuster 2', 'CLuster 3', 'CLuster 4', 'CLuster 5']
for i in range(0,5):
    plt.scatter(x[y_hc == i, 0], x[y_hc == i, 1], c = color_list[i], label = cluster_label[i], s = 60)
plt.title('Clusters of Customers (using hierarchical agglomerative)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# %% [markdown]
# ## training with 3 clusters & Visualizing it

# %%
hc_3 = AgglomerativeClustering(n_clusters= 3, linkage= 'ward', affinity= 'euclidean')
y_hc3 = hc_3.fit_predict(x)

for i in range(0,3):
    plt.scatter(x[y_hc3 == i, 0], x[y_hc3 == i, 1], c = color_list[i], label = cluster_label[i], s = 60)
plt.title('Clusters of Customers (3 clusters)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
