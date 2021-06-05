# -*- coding: utf-8 -*-
"""
Created on Wed May 26 20:37:24 2021

@author: Murillo
"""

from sklearn.mixture import GaussianMixture
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score

episodes = [2,5,15]
dict_dt = {'Mall_Customers.csv': episodes, 'buddymove_holidayiq.csv': episodes}
ix_rep = 0
dts = ['Mall_Customers.csv', 'buddymove_holidayiq.csv']
report = []

for dt, episodes in dict_dt.items():
    for ep in episodes:
        dt_ix = dts.index(dt) + 1  # Apresenta o indice 
        setup = episodes.index(ep) + 1  # Apresenta o numero do setup

        # Importing the dataset
        dataset = pd.read_csv(dt)
        X = dataset.iloc[:, [3, 4]].values
        
        
        gm = GaussianMixture(n_components=ep, random_state=0)
        clusters = gm.fit_predict(X)
        labels = clusters
        
        # Print the cluster centroids
        print(f'Results - {dt} - Setup {episodes.index(ep) + 1} ')
        davies_bouldin = davies_bouldin_score(X, clusters)
        
        metrics.silhouette = metrics.silhouette_score(X, labels, metric='euclidean')
        print(f'Silhouette Coefficient: {round(metrics.silhouette, 4)}')
        
        # Visualising the clusters
        plt.scatter(X[clusters == 0, 0], X[clusters == 0, 1], s = 50, c = 'red', label = 'Cluster 1')
        plt.scatter(X[clusters == 1, 0], X[clusters == 1, 1], s = 50, c = 'blue', label = 'Cluster 2')
        plt.scatter(X[clusters == 2, 0], X[clusters == 2, 1], s = 50, c = 'green', label = 'Cluster 3')
        plt.scatter(X[clusters == 3, 0], X[clusters == 3, 1], s = 50, c = 'cyan', label = 'Cluster 4')
        plt.scatter(X[clusters == 4, 0], X[clusters == 4, 1], s = 50, c = 'magenta', label = 'Cluster 5')
        #plt.scatter(gm.cluster_centroids_[:, 0], gm.cluster_centroids_[:, 1], s = 100, c = 'yellow', label = 'Centroids')
        plt.title('Clusters of customers')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.legend()
        plt.show()
        
        ix_rep += 1
        report.append(f'{ix_rep}  & {dt_ix} & {setup} & {round(davies_bouldin, 4)} & {round(metrics.silhouette, 4)} \\\\')

for rep in report:  # Imprime o relatório linha a linha
    print(rep)