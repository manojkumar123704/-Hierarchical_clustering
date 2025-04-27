# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 19:25:04 2025

@author: nandh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\nandh\Downloads\Mall_Customers.csv")
X= dataset.iloc[:,[3,4]].values

import scipy.cluster.hierarchy as sch


deendogram = sch.dendrogram(sch.linkage(X, method='ward'))

plt.title('dendogram')
plt.xlabel('customers')
plt.ylabel('euclidian distance')
plt.show()


from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, metric = 'euclidean' , linkage='ward')
y_hc = hc.fit_predict(X)

plt.scatter(X[y_hc == 0,0], X[y_hc == 0, 1], s=100, c='red', label = 'cluster1' )
plt.scatter(X[y_hc == 1,0], X[y_hc == 1, 1], s=100, c='blue', label = 'cluster2' )
plt.scatter(X[y_hc == 2,0], X[y_hc == 2, 1], s=100, c='green', label = 'cluster3' )
plt.scatter(X[y_hc == 3,0], X[y_hc == 3, 1], s=100, c='cyan', label = 'cluster4' )
plt.scatter(X[y_hc == 4,0], X[y_hc == 4, 1], s=100, c='magenta', label = 'cluster5' )
plt.title('clusters of customers')
plt.xlabel('annual income')
plt.ylabel('spending score(1-100)')
plt.legend()
plt.show()