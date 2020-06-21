import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import glmnet

lambdas = np.exp(np.arange(-3, 10, 0.1))
n_train = 400

def arrange_data():
    data = load_boston()
    X_train = data['data'][:n_train]
    X_test = data['data'][n_train:]
    y_train = data['target'][:n_train]
    y_test = data['target'][n_train:]
    features = data['feature_names']
    return X_train, X_test, y_train, y_test, features


def reularization_path(X_train, y_train, features, method=Lasso):
    coefs = []

    for lamb in lambdas:
        model = method(alpha=lamb).fit(X_train, y_train)
        coefs.append(model.coef_)

    coefs = np.array(coefs)
    for i in range(coefs.shape[1]):
        plt.plot(lambdas, coefs[:, i], label=features[i])
        plt.xscale("log")
    plt.legend()
    plt.show()


def cross_validation(X_train, y_train, X_test, y_test, cv=10, lasso=True):
    alpha = 1 if lasso else 0
    model = glmnet.ElasticNet(alpha=alpha, n_splits=cv, standardize=False, lambda_path=lambdas[::-1]).fit(X_train, y_train)
    print(f"best lambda for {'lasso' if lasso else 'ridge'} is {model.lambda_max_}")
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"RMSE is {rmse}")


def freedom_degrees(X_train, y_train, X_test, y_test, lasso=True):
    MSE_train, MSE_test, estimated_MSE_test = [], [], []
    if not lasso:
        u, s, v_t = np.linalg.svd(X_train)
        s_2 = np.power(s, 2)
    for lamb in lambdas:
        if lasso:
            model = Lasso(alpha=lamb).fit(X_train, y_train)
            deg = np.count_nonzero(model.coef_)
        else:
            model = Ridge(alpha=lamb).fit(X_train, y_train)
            deg = np.sum(s_2 / (s_2 + lamb))

        mse_train = mean_squared_error(model.predict(X_train), y_train)
        MSE_train.append(mse_train)
        mse_test = mean_squared_error(model.predict(X_test), y_test)
        MSE_test.append(mse_test)
        lin_reg = LinearRegression().fit(X_train, y_train)
        ls_sigma2_hat = np.sum(mean_squared_error(lin_reg.predict(X_train), y_train)) / (506-13-1)
        estimated_mse_test = mse_train + (2 * deg / n_train) * ls_sigma2_hat
        estimated_MSE_test.append(estimated_mse_test)

    plt.plot(np.log(lambdas), MSE_train, 'o', label='train')
    plt.plot(np.log(lambdas), MSE_test,label='test')
    plt.plot(np.log(lambdas), estimated_MSE_test, label='est_test')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, features = arrange_data()
    reularization_path(X_train, y_train, features)
    # reularization_path(X_train, y_train, features, method=Ridge)
    # cross_validation(X_train, y_train, X_test, y_test)
    # cross_validation(X_train, y_train, X_test, y_test, lasso=False)
    freedom_degrees(X_train, y_train, X_test, y_test, lasso=False)