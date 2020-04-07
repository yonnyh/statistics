import numpy as np


def read_data():
    with open("data_EX1.txt") as file:
        data = np.loadtxt(file)
        return data[:, 0], data[:, 1]


def permutation_test(x, y, n=1000, alpha=0.01):
    cor_coef = np.corrcoef(x, y)[1][0]
    print("Pearson correlation coefficient test is {}".format(cor_coef))

    counter = 0
    for i in range(0, n):
        y_per = np.random.permutation(y)
        counter = counter + 1 if np.corrcoef(x, y_per)[1][0] >= cor_coef else counter
        # print("cor coef {}: {}".format(i, np.corrcoef(x, y_per)[1][0]))
    pval = counter / n
    if pval < alpha:
        print("pval is {} --> reject H_0".format(pval))
    else:
        print("pval is {} --> accept H_0".format(pval))


if __name__ == '__main__':
    x, y = read_data()
    permutation_test(x,y)