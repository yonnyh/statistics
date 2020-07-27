import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ssc.DataProjection import *
from ssc.BuildAdjacency import *
from ssc.OutlierDetection import *
from ssc.BestMap import *
from ssc.SpectralClustering import *
from ssc.SparseCoefRecovery import *


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


def get_data(n, p, sigma_2):
    mean1 = np.zeros(p)
    mean2 = np.ones(p) * 5
    cov = np.eye(p) * sigma_2
    g1 = np.random.multivariate_normal(mean1, cov, n // 2)
    g2 = np.random.multivariate_normal(mean2, cov, n // 2)
    data = np.concatenate((g1, g2))
    s = np.concatenate((np.ones(n // 2), np.ones(n // 2) * 2))
    return data, s


def plot_predicted_data(Xp, Grps):
    plt.scatter(Xp.T[:, 0], Xp.T[:, 1])
    plt.show()
    l1 = np.array([Xp.T[i] for i in range(Xp.shape[1]) if Grps[i] == 1])
    l2 = np.array([Xp.T[i] for i in range(Xp.shape[1]) if Grps[i] == 2])
    plt.scatter(l1[:, 0], l1[:, 1])
    plt.scatter(l2[:, 0], l2[:, 1])
    plt.show()


def run_alg(data, s, d, cst, optm, lmbda, n_subspaces, n_edges_to_keep = 0):
    if cst == 1 and n_edges_to_keep != 0:
        n_edges_to_keep = d + 1
    display(data)
    Xp = DataProjection(data.T, d, type='NormalProj')
    display(Xp.T, data)
    CMat = SparseCoefRecovery(Xp, cst, optm, lmbda)
    # plt.imshow(CMat)
    # plt.show()

    # Make small values 0
    eps = np.finfo(float).eps
    CMat[np.abs(CMat) < eps] = 0

    CMatC, sc, OutlierIndx, Fail = OutlierDetection(CMat, s)

    if not Fail:
        CKSym = BuildAdjacency(CMatC, n_edges_to_keep)
        Grps = SpectralClustering(CKSym, n_subspaces)
        Grps = BestMap(sc, Grps)
        Missrate = float(np.sum(sc != Grps)) / sc.size
        print("Misclassification rate: {:.4f} %".format(Missrate * 100))
        plot_predicted_data(Xp, Grps)
    else:
        print("Something failed")


def main():
    data, s = get_data(n=200, p=3, sigma_2=1)
    run_alg(data, s, d=2, cst=1, optm='L1Perfect', lmbda=0.001, n_subspaces=2, n_edges_to_keep=0)


if __name__ == "__main__":
    # display()
    main()