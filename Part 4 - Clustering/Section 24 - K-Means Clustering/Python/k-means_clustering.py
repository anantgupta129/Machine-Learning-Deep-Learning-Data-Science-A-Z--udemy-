# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # K-Means Clustering

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%
dts = pd.read_csv("Mall_Customers.csv")
x = dts.iloc[:, [3, 4]].values

# %% [markdown]
# ## Using ELBOW METHOD to find optimal number of CLUSTERS

# %%
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state=42, init= 'k-means++')
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss, linestyle= '--', marker = 'o')
plt.title('The Elbow Method')
plt.xlabel('Number of CLusters')
plt.ylabel('WCSS')
plt.show()


# %%
#from the graph below we can that the optima lnumber of cluster is 5

# %% [markdown]
# ## Training K-MEANS model on dataset

# %%
kmeans = KMeans(n_clusters = 5, random_state=42, init= 'k-means++')
y_kmeans = kmeans.fit_predict(x)


# %%
print(y_kmeans)

# %% [markdown]
# ## Visualizing cluster 

# %%
color_list = ['red', 'blue', 'green', 'cyan', 'magenta']
cluster_label = ['CLuster 1', 'CLuster 2', 'CLuster 3', 'CLuster 4', 'CLuster 5']
for i in range(0,5):
    plt.scatter(x[y_kmeans == i, 0], x[y_kmeans == i, 1], c = color_list[i], label = cluster_label[i], s = 60)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c = 'yellow', label = 'centroid' ,s= 100)
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# %% [markdown]
# ## K-Means Clustering with all variables

# %%
X2 = dts.iloc[:, 1:].values
# encoding Categorical DAta
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X2[:, 0] = le.fit_transform(X2[:, 0])

#Visualizing WCSS plot for optimal number of clusters
from sklearn.cluster import KMeans
wcss_2 = []
for i in range(1,11):
    kmeans2 = KMeans(n_clusters = i, random_state=42, init= 'k-means++')
    kmeans2.fit(X2)
    wcss_2.append(kmeans2.inertia_)
plt.plot(range(1,11), wcss_2, linestyle= '--', marker = 'o')
plt.title('The Elbow Method (for all variables)')
plt.xlabel('Number of CLusters')
plt.ylabel('WCSS')
plt.show()

# %% [markdown]
# ## Training New model with all variables

# %%
kmeans2 = KMeans(n_clusters = 5, random_state=42, init= 'k-means++')
y_kmeans_2 = kmeans2.fit_predict(X2)


# %%
print(y_kmeans_2)


# %%
kmeans2.cluster_centers_

# %% [markdown]
# ## Visulaizing Cluster Plot with all Variables

# %%
color_list = ['red', 'blue', 'green', 'cyan', 'magenta']
cluster_label = ['CLuster 1', 'CLuster 2', 'CLuster 3', 'CLuster 4', 'CLuster 5']
for i in range(0,5):
    plt.scatter(x[y_kmeans_2 == i, 0], x[y_kmeans_2 == i, 1], c = color_list[i], label = cluster_label[i], s = 60)
plt.scatter(kmeans2.cluster_centers_[:, 2], kmeans2.cluster_centers_[:, 3], c = 'yellow', label = 'centroid' ,s= 100)
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


