# K-Means Clustering

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
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
    
    # Using the elbow method to find the optimal number of clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    
    # Training the K-Means model on the dataset
    kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit_predict(X)
    
    # Clustering performance evaluation
    labels = kmeans.labels_
    
    davies_bouldin = davies_bouldin_score(X, labels)
    print('Results - {dt} - Setup 01 ')
    print('Davies bouldin score:' ,davies_bouldin)
    
    metrics.silhouette = metrics.silhouette_score(X, labels, metric='euclidean')
    print('Silhouette Coefficient: ',metrics.silhouette)
    
    # Visualising the clusters
    plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 50, c = 'red', label = 'Cluster 1')
    plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 50, c = 'blue', label = 'Cluster 2')
    plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 50, c = 'green', label = 'Cluster 3')
    plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 50, c = 'cyan', label = 'Cluster 4')
    plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 50, c = 'magenta', label = 'Cluster 5')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()
    
    report_ix += 1
    report.append(f'{report_ix} & {ix_dt} & 1 & {round(davies_bouldin, 4)} & {round(metrics.silhouette, 4)} \\\\')
    
    ############################ SETUP 2 ##########################################
    
    # Training the K-Means model on the dataset
    kmeans2 = KMeans(n_clusters = 8, init = 'k-means++', random_state = 42 )
    y_kmeans = kmeans2.fit_predict(X)
    
    # Clustering performance evaluation
    labels = kmeans2.labels_
    
    davies_bouldin = davies_bouldin_score(X, labels)
    print('Results - {dt} - Setup 02 ')
    print('Davies bouldin score:' ,davies_bouldin)
    
    metrics.silhouette = metrics.silhouette_score(X, labels, metric='euclidean')
    print('Silhouette Coefficient: ',metrics.silhouette)
    
    
    # Visualising the clusters
    plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 50, c = 'red', label = 'Cluster 1')
    plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 50, c = 'blue', label = 'Cluster 2')
    plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 50, c = 'green', label = 'Cluster 3')
    plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 50, c = 'cyan', label = 'Cluster 4')
    plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 50, c = 'magenta', label = 'Cluster 5')
    plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s = 50, c = 'lime', label = 'Cluster 6')
    plt.scatter(X[y_kmeans == 6, 0], X[y_kmeans == 6, 1], s = 50, c = 'black', label = 'Cluster 7')
    plt.scatter(X[y_kmeans == 7, 0], X[y_kmeans == 7, 1], s = 50, c = 'gold', label = 'Cluster 8')
    plt.scatter(kmeans2.cluster_centers_[:, 0], kmeans2.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()
    
    report_ix += 1
    report.append(f'{report_ix} & {ix_dt} & 2 & {round(davies_bouldin, 4)} & {round(metrics.silhouette, 4)} \\\\')
    
    ############################ SETUP 3 ##########################################
    
    # Training the K-Means model on the dataset
    kmeans3 = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42 )
    y_kmeans = kmeans3.fit_predict(X)
    
    # Clustering performance evaluation
    labels = kmeans3.labels_
    
    davies_bouldin = davies_bouldin_score(X, labels)
    print('Results - {dt} - Setup 03 ')
    print('Davies bouldin score:' ,davies_bouldin)
    
    metrics.silhouette = metrics.silhouette_score(X, labels, metric='euclidean')
    print('Silhouette Coefficient: ',metrics.silhouette)
    
    
    # Visualising the clusters
    plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 50, c = 'red', label = 'Cluster 1')
    plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 50, c = 'blue', label = 'Cluster 2')
    plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 50, c = 'green', label = 'Cluster 3')
    plt.scatter(kmeans3.cluster_centers_[:, 0], kmeans3.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()
    
    report_ix += 1
    report.append(f'{report_ix} & {ix_dt} & 3 & {round(davies_bouldin, 4)} & {round(metrics.silhouette, 4)} \\\\')

for rep in report:  # Imprime o relatório linha a linha
    print(rep)
