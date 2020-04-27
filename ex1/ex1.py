import numpy as np
import matplotlib.pyplot as plt


def read_data():
    return np.genfromtxt("ex1/data_EX1.txt", delimiter='\t')


def pearson(data):
    return np.corrcoef(data[:, 0], data[:, 1])[1][0]


def hoeffding(data):
    o00, o01, o10, o11 = np.zeros(len(data)), np.zeros(len(data)), np.zeros(len(data)), np.zeros(len(data))
    for i, p in enumerate(data):
        o00[i] = np.sum(np.logical_and(data[:, 0] < p[0], data[:, 1] < p[1]))
        o01[i] = np.sum(np.logical_and(data[:, 0] < p[0], data[:, 1] > p[1]))
        o10[i] = np.sum(np.logical_and(data[:, 0] > p[0], data[:, 1] < p[1]))
        o11[i] = np.sum(np.logical_and(data[:, 0] > p[0], data[:, 1] > p[1]))
    return np.sum(np.power(o00 * o11 - o01 * o10, 2)) / len(data)**4


def likelihood_ratio(data):
    return np.power(0.625, len(data)) * np.prod(np.exp((-9 / 32) * (np.power(data[:, 0], 2) + np.power(data[:, 1], 2))) *
                    (np.exp((-15/16) * data[:, 0] * data[:, 1]) + np.exp((15/16) * data[:, 0] * data[:, 1])))


def log_likelihood_ratio(data):
    return len(data) * np.log(0.625) + np.sum(np.log(np.exp((-9 / 32) * (np.power(data[:, 0], 2) + np.power(data[:, 1], 2))) *
            (np.exp((-15 / 16) * data[:, 0] * data[:, 1]) + np.exp((15 / 16) * data[:, 0] * data[:, 1]))))


def permutation_test(data, stat, alpha, M=1000, as_boolean=False):
    s_true = stat(data)
    res = np.zeros(M)
    for i in range(M):
        y_per = np.random.permutation(data[:, 1]).reshape((-1, 1))
        data_per = np.hstack((data[:, 0].reshape((-1, 1)), y_per))
        res[i] = stat(data_per)
    pval = np.count_nonzero(res > s_true) / M
    return pval < alpha if as_boolean else pval


def bootstrap(data, stat, alpha, M=1000, as_boolean=False):
    s_true = stat(data)
    res = np.zeros(M)
    for i in range(M):
        x_per = np.random.permutation(data[:, 0]).reshape((-1, 1))
        y_per = np.random.permutation(data[:, 1]).reshape((-1, 1))
        data_per = np.hstack((x_per, y_per))
        res[i] = stat(data_per)
    pval = np.count_nonzero(res > s_true) / M
    return pval < alpha if as_boolean else pval


def sample_data(h_0=True, R=100, M=1000, alpha=0.05):
    data_size = [n for n in range(100, 1001, 100)]
    h, h_b, l, l_b = np.zeros(len(data_size)), np.zeros(len(data_size)), \
                     np.zeros(len(data_size)), np.zeros(len(data_size))

    for n_i, n in enumerate(data_size):
        print(n)
        for r in range(R):
            if r % 10 == 0: print(f"r: {r}")
            if h_0:
                data = np.random.multivariate_normal(np.zeros(2), np.identity(2), n)
            else:
                data = np.random.multivariate_normal(np.zeros(2), [[1, 0.6], [0.6, 1]], n)
                y = 2 * np.random.randint(2, size=n) - 1
                data[:, 1] = y * data[:, 1]

            if permutation_test(data, stat=hoeffding, alpha=alpha, M=M, as_boolean=True):
                h[n_i] += 1
            if bootstrap(data, stat=hoeffding, alpha=alpha, M=M, as_boolean=True):
                h_b[n_i] += 1
            if permutation_test(data, stat=log_likelihood_ratio, alpha=alpha, M=M, as_boolean=True):
                l[n_i] += 1
            if bootstrap(data, stat=log_likelihood_ratio, alpha=alpha, M=M, as_boolean=True):
                l_b[n_i] += 1

    h /= R
    h_b /= R
    l /= R
    l_b /= R

    print(h, h_b, l, l_b)

    plt.plot(data_size, h, "o--r", label="hoeffding")
    plt.plot(data_size, h_b, "o--g", label="hoeffding bootstrap")
    plt.plot(data_size, l, "o--b", label="likelihood ratio")
    plt.plot(data_size, l_b, "o--y", label="likelihood ratio bootstrap")
    plt.title("type-1 error under the null H_0" if h_0 else "power under the alternative H_1")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    data = read_data()
    alpha_q2 = 0.01
    alpha_q3 = 0.05

    # q2a
    print(f"2a: pearson coef is {pearson(data)}")
    print(f"2a: p-value of permutation test with pearson statistic is {permutation_test(data, pearson, alpha_q2)}")

    # q2b
    print(f"2b: p-value of permutation test with hoeffding statistic is {permutation_test(data, hoeffding, alpha_q2)}")

    # q2c
    print(f"2c: p-value of bootstrap with pearson statistic is {bootstrap(data, pearson, alpha_q2)}")
    print(f"2c: p-value of bootstrap with hoeffding statistic is {bootstrap(data, hoeffding, alpha_q2)}")

    # q3b
    sample_data(h_0=True, alpha=alpha_q3, R=100, M=1000)

    # q3c
    sample_data(h_0=False, alpha=alpha_q3, R=100, M=1000)
