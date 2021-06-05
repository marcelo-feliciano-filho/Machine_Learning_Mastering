# Spectral Clustering

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import SpectralClustering

datasets = ['Mall_Customers.csv', 'buddymove_holidayiq.csv']
report = []
report_ix = 0

for dt in datasets:
    ix_dt = datasets.index(dt) + 1 # Indice do dataset em questão para imprimir no relatório
    # Importing the dataset
    dataset = pd.read_csv(dt)
    X = dataset.iloc[:, [3, 4]].values
    
    # Using the elbow method to find the optimal number of clusters
    
    # Training the Spectral Clustering model on the dataset
    clustering = SpectralClustering(n_clusters=3,  assign_labels='discretize', random_state=0).fit(X)
    y_spectral = clustering.fit_predict(X)
    
    # Clustering performance evaluation
    labels = clustering.labels_
    
    davies_bouldin = davies_bouldin_score(X, labels)
    print('Results - {dt} - Setup 01 ')
    print('Davies bouldin score:' ,davies_bouldin)
    
    metrics.silhouette = metrics.silhouette_score(X, labels, metric='euclidean')
    print('Silhouette Coefficient: ',metrics.silhouette)
    
    # Visualising the clusters
    plt.scatter(X[y_spectral == 0, 0], X[y_spectral == 0, 1], s = 50, c = 'red', label = 'Cluster 1')
    plt.scatter(X[y_spectral == 1, 0], X[y_spectral == 1, 1], s = 50, c = 'blue', label = 'Cluster 2')
    plt.scatter(X[y_spectral == 2, 0], X[y_spectral == 2, 1], s = 50, c = 'green', label = 'Cluster 3')
    
    
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()
    
    report_ix += 1
    report.append(f'{report_ix} & {ix_dt} & 1 & {round(davies_bouldin, 4)} & {round(metrics.silhouette, 4)} \\\\')
    
    # ############################ SETUP 2 ##########################################
    
    clustering2 = SpectralClustering(n_clusters=5,  assign_labels='discretize', random_state=0).fit(X)
    y_spectral = clustering2.fit_predict(X)
    
    # Clustering performance evaluation
    labels = clustering2.labels_
    
    davies_bouldin = davies_bouldin_score(X, labels)
    print('Results - {dt} - Setup 02 ')
    print('Davies bouldin score:' ,davies_bouldin)
    
    metrics.silhouette = metrics.silhouette_score(X, labels, metric='euclidean')
    print('Silhouette Coefficient: ',metrics.silhouette)
    
    
    # Visualising the clusters
    plt.scatter(X[y_spectral == 0, 0], X[y_spectral == 0, 1], s = 50, c = 'red', label = 'Cluster 1')
    plt.scatter(X[y_spectral == 1, 0], X[y_spectral == 1, 1], s = 50, c = 'blue', label = 'Cluster 2')
    plt.scatter(X[y_spectral == 2, 0], X[y_spectral == 2, 1], s = 50, c = 'green', label = 'Cluster 3')
    plt.scatter(X[y_spectral == 3, 0], X[y_spectral == 3, 1], s = 50, c = 'cyan', label = 'Cluster 4')
    plt.scatter(X[y_spectral == 4, 0], X[y_spectral == 4, 1], s = 50, c = 'magenta', label = 'Cluster 5')
    
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()
    
    report_ix += 1
    report.append(f'{report_ix} & {ix_dt} & 2 & {round(davies_bouldin, 4)} & {round(metrics.silhouette, 4)} \\\\')
    
    ############################ SETUP 3 ##########################################
    
    clustering3 = SpectralClustering(n_clusters=8,  assign_labels='discretize', random_state=0).fit(X)
    y_spectral = clustering3.fit_predict(X)
    
    # Clustering performance evaluation
    labels = clustering3.labels_
    
    davies_bouldin = davies_bouldin_score(X, labels)
    print('Results - {dt} - Setup 03 ')
    print('Davies bouldin score:' ,davies_bouldin)
    
    metrics.silhouette = metrics.silhouette_score(X, labels, metric='euclidean')
    print('Silhouette Coefficient: ',metrics.silhouette)
    
    # Visualising the clusters
    plt.scatter(X[y_spectral == 0, 0], X[y_spectral == 0, 1], s = 50, c = 'red', label = 'Cluster 1')
    plt.scatter(X[y_spectral == 1, 0], X[y_spectral == 1, 1], s = 50, c = 'blue', label = 'Cluster 2')
    plt.scatter(X[y_spectral == 2, 0], X[y_spectral == 2, 1], s = 50, c = 'green', label = 'Cluster 3')
    plt.scatter(X[y_spectral == 3, 0], X[y_spectral == 3, 1], s = 50, c = 'cyan', label = 'Cluster 4')
    plt.scatter(X[y_spectral == 4, 0], X[y_spectral == 4, 1], s = 50, c = 'magenta', label = 'Cluster 5')
    plt.scatter(X[y_spectral == 5, 0], X[y_spectral == 5, 1], s = 50, c = 'lime', label = 'Cluster 6')
    plt.scatter(X[y_spectral == 6, 0], X[y_spectral == 6, 1], s = 50, c = 'black', label = 'Cluster 7')
    plt.scatter(X[y_spectral == 7, 0], X[y_spectral == 7, 1], s = 50, c = 'gold', label = 'Cluster 8')
    
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()
    
    report_ix += 1
    report.append(f'{report_ix} & {ix_dt} & 3 & {round(davies_bouldin, 4)} & {round(metrics.silhouette, 4)} \\\\')
    
for rep in report:  # Imprime o relatório linha a linha
    print(rep)
