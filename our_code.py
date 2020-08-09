import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import subspace_angles
from sklearn.cluster import KMeans

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

def get_theta(Bs):
    thetas = np.zeros((len(Bs), len(Bs)))
    for i in range(len(thetas)):
        for j in range(len(thetas)):
            thetas[i, j] = max(subspace_angles(Bs[i], Bs[j])) if i != j else 0
    return np.sum(thetas) / np.count_nonzero(thetas)


def get_data(n, p, d, sigma_2, n_subspaces, theta_coef=10**-1):
    z = np.concatenate((np.arange(1, n_subspaces+1, 1), np.random.randint(1, n_subspaces + 1, n - n_subspaces)))
    # z = np.random.randint(1, n_subspaces + 1, n)
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
        Bs = alpha * Bs + (1-alpha) * B0

    B_z = Bs[z - 1]
    x = [np.random.multivariate_normal((B_z[i] @ w[i]).reshape(-1), sigma_2 * np.eye(p)) for i in range(n)]
    return np.array(x), z, Bs


def plot_predicted_data(Xp, Grps):
    plt.scatter(Xp.T[:, 0], Xp.T[:, 1])
    plt.show()
    l1 = np.array([Xp.T[i] for i in range(Xp.shape[1]) if Grps[i] == 0])
    l2 = np.array([Xp.T[i] for i in range(Xp.shape[1]) if Grps[i] == 1])
    plt.scatter(l1[:, 0], l1[:, 1])
    plt.scatter(l2[:, 0], l2[:, 1])
    plt.show()


def run_alg(data, s, d, cst, optm, lmbda, n_subspaces, n_edges_to_keep=0, k_means=False):
    if cst == 1 and n_edges_to_keep != 0:
        n_edges_to_keep = d + 1
    # display(data)

    if k_means:
        km = KMeans(n_clusters=n_subspaces).fit(data)
        labels = km.labels_
        Grps = BestMap(s, labels + 1)

    Xp = DataProjection(data.T, d, type='NormalProj')
    # display(Xp.T, data)
    CMat = SparseCoefRecovery(Xp, cst, optm, lmbda)
    # plt.imshow(CMat)
    # plt.show()

    # Make small values 0
    eps = np.finfo(float).eps
    CMat[np.abs(CMat) < eps] = 0

    CMatC, sc, OutlierIndx, Fail = OutlierDetection(CMat, s)

    if not Fail:
        CKSym = BuildAdjacency(CMatC, n_edges_to_keep)
        Grps = SpectralClustering(CKSym, n_subspaces) + 1
        Grps = BestMap(sc, Grps)
        print(f"got: {Grps}\nexp: {sc}")
        Missrate = float(np.sum(sc != Grps)) / sc.size
        print("Misclassification rate: {:.4f} %".format(Missrate * 100))
        # plot_predicted_data(Xp, Grps)
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
    for n in range(3, 11, 1):
        for p in range(4, 8, 1):
            for d in range(-1, -5, -1):
                for theta in range(-2, 1, 1):
                    x, z, _ = get_data(n=2**n, p=2**p, d=int(2**(p+d)), theta_coef=10**theta, n_subspaces=n_subspaces, sigma_2=0.01)
                    run_alg(x, z, d=int(2**(p+d)), cst=1, optm='L1ED', lmbda=0.001, n_subspaces=n_subspaces, n_edges_to_keep=0)


    # x, z, Bs = get_data(n=100, p=3, d=2, n_subspaces=2, sigma_2=1)

    # display(x)
    # data, s = get_gaussian_data(n=200, p=3, sigma_2=1)
    # run_alg(data, s, d=2, cst=1, optm='L1Perfect', lmbda=0.001, n_subspaces=2, n_edges_to_keep=0)


if __name__ == "__main__":
    # display()
    main()
