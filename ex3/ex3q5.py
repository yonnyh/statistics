import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error as RMSE
import matplotlib.pyplot as plt
import _glmnet
import glmnet






def arrange_data():
    data = load_boston()
    X_train = data['data'][:400]
    X_test = data['data'][400:]
    y_train = data['target'][:400]
    y_test = data['target'][400:]
    features = data['feature_names']
    return X_train, X_test, y_train, y_test, features


def reularization_path(X_train, y_train, features, method=Lasso):
    coefs = []
    log_alphas = np.arange(-2, 15, 0.5)

    for log_alpha in log_alphas:
        model = method(alpha=np.exp(log_alpha)).fit(X_train, y_train)
        coefs.append(model.coef_)

    coefs = np.array(coefs)
    for i in range(coefs.shape[1]):
        plt.plot(np.exp(log_alphas), coefs[:, i], label=features[i])
        plt.xscale("log")
    plt.legend()
    plt.show()


def cross_validation(X_train, y_train, X_test, y_test, cv=10, method=LassoCV):
    # l = np.exp(np.arange(-3, 10, 0.1))
    # m = glmnet.ElasticNet(alpha=1, n_splits=5, standardize=False, lambda_path=l).fit(X_train, y_train)
    model = method(cv=10).fit(X_train, y_train)
    print(f"best alpha is {model.alpha_}")
    y_pred = model.predict(X_test)
    rmse = RMSE(y_test, y_pred, squared=False)
    print(f"RMSE is {rmse}")


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, features = arrange_data()
    # reularization_path(X_train, y_train, features)
    # reularization_path(X_train, y_train, features, method=Ridge)
    cross_validation(X_train, y_train, X_test, y_test)
    # cross_validation(X_train, y_train, X_test, y_test, method=RidgeCV)