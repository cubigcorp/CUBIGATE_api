from sklearn.cluster import KMeans
import numpy as np

def hard_split():
    """
    Split the dataset into smaller pieces so that the HW resources can handle it
    """
    pass

def soft_split(X: np.ndarray, n_cluster: int, **kargs):
    """
    Split the dataset into clusters
    """
    kmeans = KMeans(n_clusters=n_cluster, n_init='auto', **kargs).fit(X)
    return kmeans.labels_
    