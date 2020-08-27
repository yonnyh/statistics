# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.linalg import subspace_angles
# from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering as spectral_clustering
# import seaborn as sbn

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.linalg import subspace_angles
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
from fashion_mnist_master.utils import mnist_reader


from ssc.DataProjection import *
from ssc.BuildAdjacency import *
from ssc.OutlierDetection import *
from ssc.BestMap import *
from ssc.SpectralClustering import *
from ssc.SparseCoefRecovery import *


def get_theta(Bs):
    thetas = np.zeros((len(Bs), len(Bs)))
    for i in range(len(thetas)):
        for j in range(len(thetas)):
            thetas[i, j] = max(subspace_angles(Bs[i], Bs[j])) if i != j else 0
    return np.sum(thetas) / np.count_nonzero(thetas)


def get_data(n, p, d, sigma_2, n_subspaces, theta_coef=10 ** -1):
    z = np.concatenate((np.arange(0, n_subspaces, 1), np.random.randint(0, n_subspaces, n - n_subspaces)))
    w = np.random.multivariate_normal(np.zeros(d), np.eye(d), n).reshape((n, d, 1))
    Bs = np.random.uniform(-1, 1, p * d * (n_subspaces + 1)).reshape((p, d * (n_subspaces + 1)))
    Bs /= np.linalg.norm(Bs, axis=0)
    Bs = Bs.T.reshape(n_subspaces + 1, d, p).transpose(0, 2, 1)
    B0, Bs = Bs[0], Bs[1:]
    theta_max = get_theta(Bs)
    theta = theta_coef * theta_max
    alpha = 0.9
    eps = 0.01
    while abs(get_theta(Bs) - theta) > eps:
        # print(abs(get_theta(Bs) - theta))
        Bs = alpha * Bs + (1 - alpha) * B0

    Bs /= np.linalg.norm(Bs, axis=0)

    B_z = Bs[z]
    x = [np.random.multivariate_normal((B_z[i] @ w[i]).reshape(-1), sigma_2 * np.eye(p)) for i in range(n)]
    return np.array(x), z, Bs


def BestMap(L1, L2):

    L1 = L1.flatten(order='F').astype(float)
    L2 = L2.flatten(order='F').astype(float)
    if L1.size != L2.size:
        sys.exit('size(L1) must == size(L2)')
    Label1 = np.unique(L1)
    nClass1 = Label1.size
    Label2 = np.unique(L2)
    nClass2 = Label2.size
    nClass = max(nClass1, nClass2)

    # For Hungarian - Label2 are Workers, Label1 are Tasks.
    G = np.zeros([nClass, nClass]).astype(float)
    for i in range(0, nClass2):
        for j in range(0, nClass1):
            G[i, j] = np.sum(np.logical_and(L2 == Label2[i], L1 == Label1[j]))

    _, c = linear_sum_assignment(-G)
    newL2 = np.zeros(L2.shape)
    for i in range(0, nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2

def c_subspace(data, Bs, d, n_subspaces, labels):
    metric = 0
    for i in range(n_subspaces):
        b_hat = data[labels == i].T
        if data[labels == i].shape[0] > d:
            centered_data = data[labels == i] - np.mean(data[labels == i], axis=1)[: ,None]
            pca = PCA(n_components=d).fit(centered_data.T)
            b_hat = pca.transform(centered_data.T)
        metric += np.cos(max(subspace_angles(Bs[i], b_hat)))**2
    return metric

def plot_heatmap(heatmap, metric, thetas, n_range, p, d, save=False):
    assert metric in ["K-means", "C_cluster", "C_subspace"]
    sbn.heatmap(heatmap, annot=True, fmt=".2f", linewidths=.5, vmin=0,
                vmax=4 if metric == "C_subspace" else 1, xticklabels=2**n_range,
                yticklabels=10**thetas)
    title = f"{metric} heatmap for p={2**p}, d={2**(p+d)}"
    plt.title(title)
    plt.xlabel("n")
    plt.ylabel("theta coefficient")
    if save:
        plt.savefig(f"Aug27 {title}")
    plt.show()

def Kmeans(x, z, n_subspaces):
    kmeans_labels = KMeans(n_clusters=n_subspaces).fit_predict(x)
    kmeans_labels = BestMap(z, kmeans_labels)
    return accuracy_score(z, kmeans_labels)


def ssc(data, s, d, n_subspaces, cst=1, optm='Lasso', n_edges_to_keep=0):
    if cst == 1 and n_edges_to_keep != 0:
        n_edges_to_keep = d + 1
    # display(data)

    # Xp = DataProjection(data.T, d, type='NormalProj')
    # display(Xp.T, data)
    CMat = SparseCoefRecovery(data.T, cst, optm)
    # plt.imshow(CMat)
    # plt.show()

    # Make small values 0
    eps = np.finfo(float).eps
    CMat[np.abs(CMat) < eps] = 0

    CMatC, sc, OutlierIndx, Fail = OutlierDetection(CMat, s)

    if not Fail:
        CKSym = BuildAdjacency(CMatC, n_edges_to_keep)
        Grps = spectral_clustering(n_clusters=n_subspaces, affinity='precomputed').fit(CKSym).labels_
        Grps = BestMap(sc, Grps)
        # plot_predicted_data(Xp, Grps)
        return Grps.astype(int)
    else:
        print("Something failed")


def main_alg(n, p, d, theta_coef, n_subspaces, sigma_2=1, kmeans=False, cluster=True, subspace=True):
    x, z, Bs = get_data(n=2 ** n, p=2 ** p, d=int(2 ** (p + d)), theta_coef=10 ** theta_coef,
                        n_subspaces=n_subspaces, sigma_2=sigma_2)

    res_kmeans = Kmeans(x, z, n_subspaces) if kmeans else 0

    ensc_labels = ssc(x, z, d, n_subspaces)
    res_cluster = accuracy_score(z, ensc_labels) if cluster else 0
    res_subspace = c_subspace(x, Bs, int(2 ** (p + d)), n_subspaces, ensc_labels) if subspace else 0
    return res_kmeans, res_cluster, res_subspace


def run_alg(n_iters=1):
    n_subspaces = 4
    n_range = np.arange(3, 11, 1)
    theta_coef_range = np.arange(0, -3, -1, dtype=float)
    for p in range(4, 8, 1):
        for d in range(-1, -5, -1):
            cluster_heatmap = np.zeros((len(theta_coef_range), len(n_range)))
            subspace_heatmap = np.zeros((len(theta_coef_range), len(n_range)))
            kmeans_heatmap = np.zeros((len(theta_coef_range), len(n_range)))
            for i, theta_coef in enumerate(theta_coef_range):
                res_kmeans, res_cluster, res_subspace = [], [], []
                for j, n in enumerate(n_range):
                    print(f"n: {2 ** n}, p: {2 ** p}, d: {int(2 ** (p + d))}, theta_coef: {10 ** theta_coef}")
                    for iter in range(n_iters):
                        kmeans, cluster, subspace = main_alg(n=n, p=p, d=d,
                                                             theta_coef=theta_coef,
                                                             n_subspaces=n_subspaces)
                        res_kmeans.append(kmeans)
                        res_cluster.append(cluster)
                        res_subspace.append(subspace)

                    kmeans_heatmap[i, j] = np.mean(res_kmeans)
                    cluster_heatmap[i, j] = np.mean(res_cluster)
                    subspace_heatmap[i, j] = np.mean(res_subspace)

            # plot_heatmap(kmeans_heatmap, "K-means", theta_coef_range, n_range, p, d)
            plot_heatmap(cluster_heatmap, "C_cluster", theta_coef_range, n_range, p, d)
            plot_heatmap(subspace_heatmap, "C_subspace", theta_coef_range, n_range, p, d)


run_alg(n_iters=1)












def display(data1, data2=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if data1.shape[1] == 2:
        data1 = np.hstack((data1, np.zeros((data1.shape[0], 1))))
    ax.plot(data1[:, 0], data1[:, 1], data1[:, 2], '.')
    if data2 is not None:
        ax.plot(data2[:, 0], data2[:, 1], data2[:, 2], '.')
    # mean1 = np.zeros(3)
    # mean2 = np.ones(3) * 5
    # cov = np.eye(3)
    # g1 = np.random.multivariate_normal(mean1, cov, 100)
    # g2 = np.random.multivariate_normal(mean2, cov, 100)

    # ax.plot(g1[:, 0], g1[:, 1], g1[:, 2], '.')
    # ax.plot(g2[:, 0], g2[:, 1], g2[:, 2], '.')

    plt.show()


def get_gaussian_data(n, p, sigma_2):
    mean1 = np.zeros(p)
    mean2 = np.ones(p) * 5
    cov = np.eye(p) * sigma_2
    g1 = np.random.multivariate_normal(mean1, cov, n // 2)
    g2 = np.random.multivariate_normal(mean2, cov, n // 2)
    data = np.concatenate((g1, g2))
    s = np.concatenate((np.ones(n // 2) * 0, np.ones(n // 2) * 1))[:, None]
    new_data = np.hstack((data, s))
    np.random.shuffle(new_data)
    return new_data[:, :3], new_data[:, 3]




def plot_predicted_data(Xp, Grps):
    plt.scatter(Xp.T[:, 0], Xp.T[:, 1])
    plt.show()
    l1 = np.array([Xp.T[i] for i in range(Xp.shape[1]) if Grps[i] == 0])
    l2 = np.array([Xp.T[i] for i in range(Xp.shape[1]) if Grps[i] == 1])
    plt.scatter(l1[:, 0], l1[:, 1])
    plt.scatter(l2[:, 0], l2[:, 1])
    plt.show()


def c_cluster(true_tags, tags):
    # print(f"got: {tags}\nexp: {true_tags}")
    true_rate = float(np.sum(true_tags == tags)) / true_tags.size
    print("True classification rate: {:.4f} %".format(true_rate * 100))
    return true_rate

#
# def c_subspace(data, Bs, d, n_subspaces, Grps):
#     metric = 0
#     for i in range(n_subspaces):
#         # B_k_hat = DataProjection(data[Grps == i].T, d, type='NormalProj')
#         metric += np.cos(max(subspace_angles(Bs[i], data[Grps == i].T)))**2
#     return metric


def k_means(data, true_labels, n_subspaces):
    Grps = KMeans(n_clusters=n_subspaces).fit(data).labels_
    Grps = BestMap(true_labels, Grps)
    miss_rate = float(np.sum(true_labels == Grps)) / true_labels.size
    print("K-Means True classification rate: {:.4f} %".format(miss_rate * 100))
    return miss_rate


def ssc(data, s, d, cst, optm, lmbda, n_subspaces, n_edges_to_keep=0):
    if cst == 1 and n_edges_to_keep != 0:
        n_edges_to_keep = d + 1
    # display(data)

    # Xp = DataProjection(data.T, d, type='NormalProj')
    # display(Xp.T, data)
    CMat = SparseCoefRecovery(data.T, cst, optm, lmbda)
    # plt.imshow(CMat)
    # plt.show()

    # Make small values 0
    eps = np.finfo(float).eps
    CMat[np.abs(CMat) < eps] = 0

    CMatC, sc, OutlierIndx, Fail = OutlierDetection(CMat, s)

    if not Fail:
        CKSym = BuildAdjacency(CMatC, n_edges_to_keep)
        Grps = spectral_clustering(n_clusters=n_subspaces, affinity='precomputed').fit(CKSym).labels_
        Grps = BestMap(sc, Grps)
        # plot_predicted_data(Xp, Grps)
        return Grps.astype(int)
    else:
        print("Something failed")


def surface(x, z, Bs):
    cross = lambda x, y: np.array([x[1] * y[2] - x[2] * y[1], -(x[0] * y[2] - x[2] * y[0]), x[0] * y[1] - x[1] * y[0]])
    plt3d = plt.figure().gca(projection='3d')
    xx, yy = np.meshgrid(range(-5,5), range(-5,5))
    norm0 = cross(Bs[0][:, 0], Bs[0][:, 1])
    norm1 = cross(Bs[1][:, 0], Bs[1][:, 1])
    point0 = x[np.where(z==1)[0][0]]
    point1 = x[np.where(z == 2)[0][0]]
    d = -point0.dot(norm0)
    zz = (-norm0[0] * xx - norm0[1] * yy - d) * 1. /norm0[2]
    plt3d.plot_surface(xx, yy, zz, alpha=0.2)
    d = -point1.dot(norm1)
    zz = (-norm1[0] * xx - norm1[1] * yy - d) * 1. / norm1[2]
    plt3d.plot_surface(xx, yy, zz, alpha=0.2)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    l1 = np.array([x[i] for i in range(x.shape[0]) if z[i] == 1])
    l2 = np.array([x[i] for i in range(x.shape[0]) if z[i] == 2])
    # l3 = np.array([x[i] for i in range(x.shape[0]) if z[i] == 3])
    # l4 = np.array([x[i] for i in range(x.shape[0]) if z[i] == 4])
    plt3d.plot(l1[:, 0], l1[:, 1], l1[:, 2], '.')
    plt3d.plot(l2[:, 0], l2[:, 1], l2[:, 2], '.')
    # ax.plot(l3[:, 0], l3[:, 1], l3[:, 2], '.')
    # ax.plot(l4[:, 0], l4[:, 1], l4[:, 2], '.')
    plt.show()


def main():
    n_subspaces = 4
    n_range = np.arange(3, 11, 1)
    theta_coef_range = np.arange(0, -3, -1, dtype=float)
    for p in range(4, 8, 1):
        for d in range(-1, -5, -1):
            cluster_heatmap = np.zeros((len(theta_coef_range), len(n_range)))
            subspace_heatmap = np.zeros((len(theta_coef_range), len(n_range)))
            kmeans_heatmap = np.zeros((len(theta_coef_range), len(n_range)))
            thetas = []
            for i, theta_coef in enumerate(theta_coef_range):
                theta_arr = []
                # n_iter = 3
                # res_kmeans, res_cluster, res_subspace = [], [], []
                for j, n in enumerate(n_range):
                    print(f"n: {2 ** n}, p: {2 ** p}, d: {int(2 ** (p + d))}, theta_coef: {10 ** theta_coef}")
                    x, z, Bs, theta = get_data(n=2 ** n, p=2 ** p, d=int(2 ** (p + d)), theta_coef=10 ** theta_coef,
                                               n_subspaces=n_subspaces, sigma_2=1)
                    theta_arr.append(theta)

                    # kmeans_labels = KMeans(n_clusters=n_subspaces).fit(x).labels_
                    # kmeans_labels = BestMap(z, kmeans_labels)
                    # kmeans_heatmap[i, j] = c_cluster(z, kmeans_labels)

                    # model = ElasticNetSubspaceClustering(n_clusters=n_subspaces, algorithm='lasso_lars', gamma=50).fit(
                    #     x)
                    # labels_permuted = BestMap(z, model.labels_)
                    # cluster_heatmap[i, j] = c_cluster(z, labels_permuted)
                    # subspace_heatmap[i, j] = c_subspace(x, Bs, int(2 ** (p + d)), n_subspaces, labels_permuted)
                thetas.append(np.mean(theta_arr))
            # plot_heatmap(kmeans_heatmap, "K-means", thetas)
            # plot_heatmap(cluster_heatmap, "C_cluster", thetas)
            # plot_heatmap(subspace_heatmap, "C_subspace", thetas)
    # x, z, Bs = get_data(n=100, p=3, d=2, n_subspaces=2, sigma_2=1)

    # display(x)
    # data, s = get_gaussian_data(n=200, p=3, sigma_2=1)
    # run_alg(data, s, d=2, cst=1, optm='L1Perfect', lmbda=0.001, n_subspaces=2, n_edges_to_keep=0)


# if __name__ == "__main__":
    # display()
    # main()
