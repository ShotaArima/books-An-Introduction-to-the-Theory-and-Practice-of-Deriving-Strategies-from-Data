import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class PolynomialRegressor:
    def __init__(self, degree: int):
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree, include_bias=True)
        self.model = LinearRegression()

    def fit(self, x: np.ndarray, y: np.ndarray):
        X_poly = self.poly.fit_transform(x.reshape(-1, 1))
        self.model.fit(X_poly, y)

    def predict(self, x: np.ndarray):
        X_poly = self.poly.transform(x.reshape(-1, 1))
        return self.model.predict(X_poly)