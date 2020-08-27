import numpy as np
from fashion_mnist_master.utils import mnist_reader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.linalg import subspace_angles
from sklearn.cluster import KMeans
from ssc.BestMap import *

labels = ['T-shirt/top','Trouser','Pullover','Dress','Coat','sandal','Shirt',
          'Sneaker','bag','Ankle Boot']
n_clusters = 10


def get_mnist_data():
    data = mnist_reader.load_mnist('fashion_mnist_master/data/fashion', kind='train')
    X = data[0].astype(float)
    y = data[1].astype(float)

    for i in range(10):
        X[y == i] -= np.mean(X[y == i], axis=0)

    return X, y


def plot_projected_data(X, y):
    pca = PCA(n_components=2)
    X = pca.fit(X).transform(X)

    plt.figure(figsize=(10, 7))
    for i in range(n_clusters):
        plt.scatter(X[y == i, 0], X[y == i, 1], alpha=0.1, label=labels[i])
    plt.title("Fashion Mnist - PCA with 2 components")
    plt.legend()
    plt.show()


def plot_angles(X, y):
    n_pairs = 5000
    same = np.array([X[np.random.choice(np.where(y == np.random.choice(n_clusters))[0], size=2)] for _ in range(n_pairs)])
    labels = np.array([np.random.choice(n_clusters, size=2, replace=False) for _ in range(n_pairs)]).flatten()
    different = np.array([X[np.random.choice(np.where(y == label)[0])] for label in labels]).reshape((n_pairs, 2, 784))
    same_angles = [np.degrees(max(subspace_angles(pair[0][:, None], pair[1][:, None]))) for pair in same]
    diff_angles = [np.degrees(max(subspace_angles(pair[0][:, None], pair[1][:, None]))) for pair in different]

    plt.figure(figsize=(10, 7))
    plt.hist(same_angles, alpha=0.5, label="same class", bins=100, density=True)
    plt.hist(diff_angles, alpha=0.5, label="different classes", bins=100, density=True)
    plt.title("Histogram of angles between pairs of data-points")
    plt.legend()
    plt.show()


def pca_by_classes(X, y):
    n_components = 200

    plt.figure(figsize=(10, 7))
    for i in range(n_clusters):
        X_by_y = X[y == i]
        pca = PCA(n_components=n_components).fit(X_by_y)
        plt.plot(np.cumsum(pca.explained_variance_ratio_), label=labels[i])
    plt.title(f"explained variance ratio of top {n_components} components, by class")
    plt.xlabel("number of components")
    plt.ylabel("explained variance ratio")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 7))
    pca_all = PCA(n_components=n_components).fit(X)
    plt.plot(np.cumsum(pca_all.explained_variance_ratio_))
    plt.title(f"explained variance ratio of top {n_components} components, for all classes")
    plt.xlabel("number of components")
    plt.ylabel("explained variance ratio")
    plt.show()


X, y = get_mnist_data()
# plot_projected_data(X, y)
plot_angles(X, y)
# pca_by_classes(X, y)