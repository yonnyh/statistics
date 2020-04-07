import numpy as np
import matplotlib.pyplot as plt


def read_data():
    with open("data_EX1.txt") as file:
        data = np.loadtxt(file)
        return data


def independent_data(size=1000):
    np.random.seed(0)
    x = np.random.randn(size).reshape((-1, 1))
    np.random.seed(1)
    y = np.random.randn(size).reshape((-1, 1))
    return np.hstack((x, y))


def circle_data(size=1000, rad=3):
    theta = np.linspace(0, 2*np.pi, size)
    x = (np.cos(theta) * rad).reshape((-1, 1))
    y = (np.sin(theta) * rad).reshape((-1, 1))
    return np.hstack((x, y))


def pearson(data):
    return np.corrcoef(data[:, 0], data[:, 1])[1][0]


def permutation_test(data, stat, n=1000, alpha=0.01):
    s = stat(data)
    print("basic test is {}".format(s))

    counter = 0
    for i in range(0, n):
        y_per = np.random.permutation(data[:, 1]).reshape((-1, 1))
        data_per = np.hstack((data[:, 0].reshape((-1,1)), y_per))
        counter = counter + 1 if stat(data_per) > s else counter
        # print(f"{i}: {stat(data_per)}")
    pval = counter / n
    if pval < alpha:
        print("pval is {} --> reject H_0".format(pval))
    else:
        print("pval is {} --> accept H_0".format(pval))


def hoeffding(data):
    h_n = 0
    for p in data:
        o1 = data[data[:, 0] > p[0]]
        o0 = data[data[:, 0] < p[0]]
        o00 = o0[o0[:, 1] < p[1]]
        o01 = o0[o0[:, 1] > p[1]]
        o10 = o1[o1[:, 1] < p[1]]
        o11 = o1[o1[:, 1] > p[1]]
        h_n += ((len(o00) * len(o11)) - (len(o01) * len(o10)))**2
    return h_n / len(data)**4

if __name__ == '__main__':
    data = read_data()
    # permutation_test(data, pearson)
    permutation_test(data, hoeffding)