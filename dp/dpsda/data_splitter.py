from faiss import Kmeans
import numpy as np
from typing import Optional

def hard_split():
    """
    Split the dataset into smaller pieces so that the HW resources can handle it
    """
    pass

def soft_split(X: np.ndarray, num_sub_label: int, plot_result: bool = True, folder: Optional[str] = None, **kargs):
    """
    Split the dataset into clusters
    """
    X = X.reshape((len(X), -1))
    kmeans = Kmeans(X.shape[-1], num_sub_label, **kargs)
    kmeans.train(X)
    _, sub_labels = kmeans.index.search(X, 1)
    sub_labels = sub_labels.flatten()

    if plot_result:
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        colors = plt.cm.rainbow(np.linspace(0, 1, num_sub_label))
        pca = PCA(n_components=num_sub_label).fit(X)
        reduced_X = pca.transform(X)
        centers = pca.transform(kmeans.centroids)
        plt.figure()
        for label in range(num_sub_label):
            plt.scatter(reduced_X[sub_labels == label, 0], reduced_X[sub_labels == label, 1], color=colors[label], label=f'Sub-label: {label}')
            plt.scatter(centers[label, 0], centers[label, 1], color=colors[label], marker='*', s=200, label=f'Centroid: {label}')
        plt.title(f'Clustring into {num_sub_label} sub-labels')
        plt.xlabel('PCA component 1')
        plt.ylabel('PCA component 2')
        plt.legend()
        plt.savefig(f'{folder}/cluster.png')
        plt.close()
    return sub_labels
    