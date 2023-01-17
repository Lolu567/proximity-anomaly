# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:57:21 2023

@author: adeoluwa adelana
student id:21058791
"""


# Import packages
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans


# Import the stock dataset
df = pd.read_excel('C:/Users/USER/Downloads/sku_data.xlsx')
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()
print(df)

# queries used to evaluate data
df.head() # the first 4 flowers are setosa
df.info()
df.describe()
df.columns
df.isnull().sum() # there are no null values in the data


# Define the attributes to be used for the y-axis
attributes = ["Expire date","Outbound number","Total outbound","Pal grossweight","Pal height","Units per pal"]

# Create a new column in the dataframe to store the synthetic feature
df["Synthetic_Y"] = 0

# Loop through the attributes and calculate the synthetic feature
for attribute in attributes:
    df["Synthetic_Y"] += df[attribute]

# Scale the synthetic feature
scaler = MinMaxScaler()
df["Synthetic_Y"] = scaler.fit_transform(df["Synthetic_Y"].values.reshape(-1, 1))

df_sc = df[['Unitprice', 'Synthetic_Y']].dropna()
print(np.isnan(df_sc).any())
print(np.isinf(df_sc).any())


# Visualising the data to see if the mis-labelled cases look like anomalies
sns.set()
sns.scatterplot(data=df_sc, x='Unitprice', y='Synthetic_Y')
plt.scatter(x=0.0137, y=1.000000, marker='X')
plt.scatter(x=0.0000, y=0.991294, marker='X')
plt.xlabel('Unitprice')
plt.ylabel('Synthetic_Y')
plt.title('Unitprice v Synthetic_Y')
plt.show()

# Using the elbow method in K means clustering to find the optimal number of clusters 
   # WCSS is the sum of the squared distances from each point in a cluster to the centre of the cluster.
   # init refers to the initial cluster centres. k-means ++ speeds up convergence.
   # 3 looks like a reasonable number of clusters"""


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)  # Firstly call the algorithm
    kmeans.fit(df_sc)  # fit is always used to train an algorithm
    wcss.append(kmeans.inertia_)  # inertia_ gives us the wcss value for each cluster.
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method',fontsize=20)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 2).fit(df_sc)


# Test 1 - test to see if the anomalies are far from the cluster centroids


# Obtain predictions and calculate distance from cluster centroid
df_sc_clusters = kmeans.predict(df_sc)
df_sc_clusters_centers = kmeans.cluster_centers_

dist = [np.linalg.norm(x-y) for x, y in zip(df_sc.values, df_sc_clusters_centers[df_sc_clusters])]

print(df_sc_clusters)
print(dist)


# Create fraud predictions based on outliers on clusters

km_y_pred = np.array(dist)
km_y_pred[dist >= np.percentile(dist, 95)] = 1
km_y_pred[dist < np.percentile(dist, 95)] = 0

# The anomalies flagged using distances from the centroid are not the mis-labelled cases. As you will see
# in test 2 this is because one of the three clusters contain only the mis-labelled cases.


# Test 2 - Testing to see if one of the clusters contain only the mis-labelled cases


# Versicolor dataframe with the clusters
df_clus = pd.concat([df_sc,
                        pd.DataFrame(df_sc_clusters,columns=['Clusters'])],axis=1)

# We can see that one of the clusters contain only the mis-labelled cases

plt.figure()
sns.scatterplot(data=df_clus, x='Unitprice', y='Synthetic_Y', hue='Clusters', palette='deep')
plt.xlabel('Unitprice')
plt.ylabel('Synthetic_Y')
plt.legend( loc='lower right')
plt.title('Unitprice v Synthetic_Y')
plt.show()



# Density based clustering method (DBSCAN) to detect anomalies.

#get the data
X = df_sc

# Define the DBSCAN model
dbscan = DBSCAN(eps=5, min_samples=5)

# Fit the model to the data
dbscan.fit(X)

# Get the labels for each data point
labels = dbscan.labels_

# Count the number of points in each cluster
unique_labels = set(labels)
print(unique_labels)

# Identify the points that are not part of any cluster (anomalies)
anomalies = [i for i, label in enumerate(labels) if label == -1]
print(anomalies)

# Plot the data points with the cluster labels
plt.scatter(X["Unitprice"], X["Synthetic_Y"], c=dbscan.labels_)
plt.title("DBSCAN Clusters")
plt.show()



