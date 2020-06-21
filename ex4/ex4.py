import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import logistic
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from glmnet import LogitNet

p = 100
n = 200

methods = ['knn', 'lasso', 'ridge']
x_label_dict = {'knn': 'k', 'lasso': 'num of non-zero coefficients', 'ridge': 'log lambda'}


def get_data():
    np.random.seed(0)
    beta0 = 2
    beta = np.array([1] * 10 + [-1] * 10 + [0] * 80)[None, :]
    X = np.random.uniform(0, 1, p*n).reshape((p, n))
    f_true = logistic.cdf(beta0 + beta @ X)[0]
    return X, f_true


def reorder_dict(dict):
    dict = {k: np.mean(v) for k, v in dict.items()}
    idx = np.argsort(np.array(list(dict.keys())))
    x_axis = np.array(list(dict.keys()))[idx]
    y_axis = np.array(list(dict.values()))[idx]
    return x_axis, y_axis


def avg_graphs(X, f_true, method):
    assert method in methods
    n_simulations = 10
    probs = [[1 - ff, ff] for ff in f_true]
    avg_square_bias, avg_variance, avg_mse = defaultdict(list), defaultdict(list), defaultdict(list)
    loop_list = np.arange(1, 101, 1) if method == 'knn' else np.exp(np.arange(-3, 7, 0.1))
    for simulation in range(n_simulations):
        y = [np.random.choice(2, p=prob) for prob in probs]
        for l in loop_list:
            if method == 'knn':
                model = KNeighborsClassifier(n_neighbors=l).fit(X.T, y)
            elif method == 'lasso':
                model = LogisticRegression(penalty='l1', solver='liblinear', C=l).fit(X.T, y)
            else:
                model = LogitNet(alpha=0, lambda_path=[l]).fit(X.T, y)
            f_hat = model.predict_proba(X.T)[:, 1]
            x_val = l if method == 'knn' else (np.count_nonzero(model.coef_) if method == 'lasso' else np.log(l))
            avg_square_bias[x_val].append(mean_squared_error(f_true, f_hat.T))
            avg_variance[x_val].append(np.mean(np.var(f_hat)))
            avg_mse[x_val].append(mean_squared_error(f_hat.T, y))

    asb_x, asb_y = reorder_dict(avg_square_bias)
    av_x, av_y = reorder_dict(avg_variance)
    am_x, am_y = reorder_dict(avg_mse)

    plt.plot(asb_x, asb_y, label='avg_square_bias')
    plt.plot(av_x, av_y, label='avg_variance')
    plt.plot(am_x, am_y, label='avg_MSE')
    plt.title(f"graphs for {method} predictor")
    plt.xlabel('k' if method == 'knn' else ('num of non-zero coefficients' if method == 'lasso' else 'log lambda'))
    plt.legend()
    plt.show()

    print(f"for {method}, the optimal MSE is {np.min(am_y)} and we get it when {x_label_dict[method]} is {am_x[np.argmin(am_y)]}")





if __name__ == '__main__':
    X, f_true = get_data()
    avg_graphs(X, f_true, 'knn')
    # avg_graphs(X, f_true, 'lasso')
    # avg_graphs(X, f_true, 'ridge')
