# Hierarchical Clustering

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score


datasets = ['Mall_Customers.csv', 'buddymove_holidayiq.csv']
report = []
report_ix = 0

for dt in datasets:
    ix_dt = datasets.index(dt) + 1  # Indice do dataset em questão para imprimir no relatório
    # Importing the dataset
    dataset = pd.read_csv(dt)
    X = dataset.iloc[:, [3, 4]].values
    
    # Using the dendrogram to find the optimal number of clusters
    dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
    plt.title('Dendrogram')
    plt.xlabel('Customers')
    plt.ylabel('Euclidean distances')
    plt.show()
    
    # Training the Hierarchical Clustering model on the dataset
    hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
    y_hc = hc.fit_predict(X)
    
    
    # Clustering performance evaluation
    labels = hc.labels_
    
    davies_bouldin = davies_bouldin_score(X, labels)
    print('Results - {dt} - Setup 01 ')
    print('Davies bouldin score:' ,davies_bouldin)
    
    metrics.silhouette = metrics.silhouette_score(X, labels, metric='euclidean')
    print('Silhouette Coefficient: ',metrics.silhouette)
    
    
    # Visualising the clusters
    plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 50, c = 'red', label = 'Cluster 1')
    plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 50, c = 'blue', label = 'Cluster 2')
    plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 50, c = 'green', label = 'Cluster 3')
    plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 50, c = 'cyan', label = 'Cluster 4')
    plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 50, c = 'magenta', label = 'Cluster 5')
    
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()
    
    report_ix += 1
    report.append(f'{report_ix} & {ix_dt} & 1 & {round(davies_bouldin, 4)} & {round(metrics.silhouette, 4)} \\\\')
    
    ############################ SETUP 2 ##########################################
    
    # Training the K-Means model on the dataset
    hc = AgglomerativeClustering(n_clusters = 8, affinity = 'euclidean', linkage = 'ward')
    y_hc = hc.fit_predict(X)
    
    # Clustering performance evaluation
    labels = hc.labels_
    
    davies_bouldin = davies_bouldin_score(X, labels)
    print('Results - {dt} - Setup 02 ')
    print('Davies bouldin score:' ,davies_bouldin)
    
    metrics.silhouette = metrics.silhouette_score(X, labels, metric='euclidean')
    print('Silhouette Coefficient: ',metrics.silhouette)
    
    
    # Visualising the clusters
    plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 50, c = 'red', label = 'Cluster 1')
    plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 50, c = 'blue', label = 'Cluster 2')
    plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 50, c = 'green', label = 'Cluster 3')
    plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 50, c = 'cyan', label = 'Cluster 4')
    plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 50, c = 'magenta', label = 'Cluster 5')
    plt.scatter(X[y_hc == 5, 0], X[y_hc == 5, 1], s = 50, c = 'lime', label = 'Cluster 6')
    plt.scatter(X[y_hc == 6, 0], X[y_hc == 6, 1], s = 50, c = 'black', label = 'Cluster 7')
    plt.scatter(X[y_hc == 7, 0], X[y_hc == 7, 1], s = 50, c = 'gold', label = 'Cluster 8')
    
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()
    
    report_ix += 1
    report.append(f'{report_ix} & {ix_dt} & 2 & {round(davies_bouldin, 4)} & {round(metrics.silhouette, 4)} \\\\')
    ############################ SETUP 3 ##########################################
    
    # Training the K-Means model on the dataset
    hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
    y_hc = hc.fit_predict(X)
    
    # Clustering performance evaluation
    labels = hc.labels_
    
    davies_bouldin = davies_bouldin_score(X, labels)
    print('Results - {dt} - Setup 03 ')
    print('Davies bouldin score:' ,davies_bouldin)
    
    metrics.silhouette = metrics.silhouette_score(X, labels, metric='euclidean')
    print('Silhouette Coefficient: ',metrics.silhouette)
    
    
    # Visualising the clusters
    plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 50, c = 'red', label = 'Cluster 1')
    plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 50, c = 'blue', label = 'Cluster 2')
    plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 50, c = 'green', label = 'Cluster 3')
    
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()
    
    report_ix += 1
    report.append(f'{report_ix} & {ix_dt} & 3 & {round(davies_bouldin, 4)} & {round(metrics.silhouette, 4)} \\\\')

for rep in report:  # Imprime o relatório linha a linha
    print(rep)
    