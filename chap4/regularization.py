import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso

class LassoPolynomialRegressor:
    def __init__(self, degree: int, alpha: float):
        self.degree = degree
        self.alpha = alpha
        self.poly = PolynomialFeatures(degree=degree, include_bias=True)
        self.scaler = StandardScaler()
        self.model = Lasso(alpha=alpha, max_iter=10000)

    def fit(self, x, y):
        X_poly = self.poly.fit_transform(x.reshape(-1, 1))
        X_poly_std = self.scaler.fit_transform(X_poly)
        self.model.fit(X_poly_std, y)

    def predict(self, x):
        X_poly = self.poly.transform(x.reshape(-1, 1))
        X_poly_std = self.scaler.transform(X_poly)
        return self.model.predict(X_poly_std)


def lasso_path(x, y, degree=9, lambdas=np.logspace(-4, 2, 80)):
    """Lasso 解パスを返す"""
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    scaler = StandardScaler()

    X_poly = poly.fit_transform(x.reshape(-1, 1))
    X_poly_std = scaler.fit_transform(X_poly)

    coefs = []

    for lam in lambdas:
        model = Lasso(alpha=lam, max_iter=20000)
        model.fit(X_poly_std, y)
        coefs.append(model.coef_)

    return np.array(coefs), np.log(lambdas), poly.get_feature_names_out()
