#clustering.py

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from nr14_data_analyser.preprocessor import Cleaner
 

class Clustering:
    def __init__(self, data):
        self.data = data

    def kmeans_clustering(self, X, n_clusters=3, random_state=42):
        model = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = model.fit_predict(X)
        return labels, model

    def dbscan_clustering(self, X, eps=0.5, min_samples=5):
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            print("Warning: DBSCAN produced fewer than 2 clusters. Metrics may be invalid.")
        
        return labels, model


    def agglomerative_clustering(self, X, n_clusters=3, linkage_method="ward"):
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        labels = model.fit_predict(X)
        return labels, model

    def hierarchical_clustering(self, X, method="ward"):
        from scipy.cluster.hierarchy import linkage
        Z = linkage(X, method=method)
        return Z

    def plot_dendrogram(self, Z, truncate_mode=None, p=12):
        plt.figure(figsize=(10, 6))
        dendrogram(Z, truncate_mode=truncate_mode, p=p)
        plt.xlabel("Samples")
        plt.ylabel("Distance")
        plt.title("Hierarchical Clustering Dendrogram")
        return plt.gcf()