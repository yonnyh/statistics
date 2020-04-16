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


def likelihood_ratio(data):
    return np.prod(np.exp(((9 / 32) * (data[:, 0]**2 + data[:, 1]**2)) - ((15/16) * data[:, 0] * data[:, 1] * (np.e + np.e ** -1))))


def basic_stat(stat, data):
    s = stat(data)
    print("basic test is {}".format(s))
    return s


def permutation_test(data, stat, n=100, alpha=0.01):
    s = basic_stat(stat, data)

    counter = 0
    for i in range(0, n):
        y_per = np.random.permutation(data[:, 1]).reshape((-1, 1))
        data_per = np.hstack((data[:, 0].reshape((-1, 1)), y_per))
        counter = counter + 1 if stat(data_per) > s else counter
        # print(f"{i}: {stat(data_per)}")
    pval = counter / n
    return pval
    # if pval < alpha:
    #     print(f"pval is {pval} --> reject H_0")
    # else:
    #     print(f"pval is {pval} --> accept H_0")


def bootstrap(data, stat, n=1000):
    # s = basic_stat(stat, data)
    stats = []
    for i in range(n):
        x_per = np.random.permutation(data[:, 0]).reshape((-1, 1))
        y_per = np.random.permutation(data[:, 1]).reshape((-1, 1))
        data_per = np.hstack((x_per, y_per))
        stats.append(stat(data_per))
    # print(f"the statistic value after bootstrap is {np.mean(stats)}, i.e the difference is {np.abs(s-np.mean(stats))}")
    return np.mean(stats)



def q3b(independent=True):
    mean = [0,0]
    cov = [[1, 0.6], [0.6, 1]]
    data_size = [n for n in range(100, 1001, 100)]
    hoef, hoef_boot, lr, lr_boot = [], [], [], []

    for n in [1000]:
        print(n)
        if independent:
            data = np.random.multivariate_normal(mean, [[1.6, 0], [0, 0.4]], n)
        else:
            # data = np.random.multivariate_normal(mean, cov, n)
            data = read_data()
        y = 2 * np.random.randint(2, size=n) - 1
        data[:, 1] = y * data[:, 1]
        basic_stat(hoeffding, data)
        basic_stat(likelihood_ratio, data)
        lr_boot.append(bootstrap(data, likelihood_ratio))
        hoef.append(permutation_test(data, hoeffding))
        hoef_boot.append(bootstrap(data, hoeffding))
        lr.append(permutation_test(data, likelihood_ratio))
    # plt.plot(data_size, hoef, "o-r", label="hoeffding")
    # plt.plot(data_size, hoef_boot, "o--g", label="hoeffding bootstrap")
    # plt.plot(data_size, hoef, "o-b", label="likelihood ratio")
    # plt.plot(data_size, hoef, "o--y", label="likelihood ratio bootstrap")
    print(hoef)
    print(hoef_boot)
    print(lr)
    print(lr_boot)
    # plt.grid(b=True, which='major', color='#666666', linestyle='-')
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    data = read_data()
    # permutation_test(data, pearson)
    # permutation_test(data, hoeffding)
    # bootstrap(data, pearson)
    # bootstrap(data, hoeffding)
    q3b()
