# -*- coding: utf-8 -*-
"""
Created on Thu May 27 20:20:10 2021

@author: Bruno Souza
"""

# MeanShift

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import MeanShift
from itertools import cycle
        

episodes = [2,4,8]
dict_dt = {'Mall_Customers.csv': episodes, 'buddymove_holidayiq.csv': episodes}
report = []
dts = ['Mall_Customers.csv', 'buddymove_holidayiq.csv']
ix_rep = 0

for dt, episodes in dict_dt.items():
    for ep in episodes:
        dt_ix = dts.index(dt) + 1  # Apresenta o indice 
        setup = episodes.index(ep) + 1  # Apresenta o numero do setup
        
        # Importing the dataset
        dataset = pd.read_csv(dt)
        X = dataset.iloc[:, [3, 4]].values
        
        # Using the elbow method to find the optimal number of clusters
        # Training the K-Means model on the dataset
        ms = MeanShift(bandwidth=ep).fit(X)
        labels = ms.labels_
        # Clustering performance evaluation
    
        davies_bouldin = davies_bouldin_score(X, labels)
        print(f'Results - {dt} - Setup {episodes.index(ep) + 1} ')
        print(f'Davies bouldin score: {round(davies_bouldin, 4)}')
        
        metrics.silhouette = metrics.silhouette_score(X, labels, metric='euclidean')
        print(f'Silhouette Coefficient: {round(metrics.silhouette, 4)}')
        
        cluster_centers = ms.cluster_centers_
        # Visualising the clusters
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        
        print(f"number of estimated clusters : {n_clusters_}")
        
        # #############################################################################
        # Plot result
        
        plt.figure(1)
        plt.clf()
        
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters_), colors):
            my_members = labels == k
            cluster_center = cluster_centers[k]
            plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
            plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=14)
        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()
        
        ix_rep += 1
        report.append(f'{ix_rep}  & {dt_ix} & {setup} & {round(davies_bouldin, 4)} & {round(metrics.silhouette, 4)} \\\\')

for rep in report:  # Imprime o relatório linha a linha
    print(rep)
