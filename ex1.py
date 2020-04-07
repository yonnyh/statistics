import numpy as np
import matplotlib.pyplot as plt
# from mlxtend.evaluate import permutation_test
# from sklearn.model_selection import permutation_test_score
# from scipy.stats import zscore


def main():
    with open("data_EX1.txt") as file:
        data = np.loadtxt(file)
        # plt.plot(data[:, 0], data[:, 1], ".")
        # plt.show()
        # pearson = np.corrcoef(data[:, 0], data[:, 1])
        # zscore = z_score(data[:, 0], data[:, 1])
        pval = permutation_test(data[:, 0], data[:, 1], 100, z_score)
        print(pval)

    return


def permutation_test(x, y, n, stat):
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    t = stat(x, y)
    sum_of_eccentrics = 0
    for i in range(n):
        perm = np.random.permutation(y)
        t_i = stat(x, perm)
        if t_i > t:
            sum_of_eccentrics += 1
    return sum_of_eccentrics / x.size


def z_score(x, y):
    return (np.mean(x) - np.mean(y)) / \
           np.sqrt(((np.std(x)**2) / x.size) + ((np.std(y)**2) / y.size))


if __name__ == '__main__':
    main()
