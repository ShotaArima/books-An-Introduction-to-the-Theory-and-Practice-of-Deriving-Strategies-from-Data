import numpy as np
from sklearn.model_selection import train_test_split
from evaluator import mae, rmse
from polynomial_model import PolynomialRegressor

class PolynomialExperiment:
    def __init__(self, degrees=range(1, 10), test_size=0.3, random_state=0):
        self.degrees = degrees
        self.test_size = test_size
        self.random_state = random_state
        self.results = {}

    def run(self, x, y):
        Xtr, Xte, ytr, yte = train_test_split(
            x, y, test_size=self.test_size, random_state=self.random_state
        )

        mae_tr, mae_te, rmse_tr, rmse_te = [], [], [], []

        # 各次数ごとの実験
        for deg in self.degrees:
            model = PolynomialRegressor(degree=deg)
            model.fit(Xtr, ytr)

            ytr_hat = model.predict(Xtr)
            yte_hat = model.predict(Xte)

            mae_tr.append(mae(ytr, ytr_hat))
            mae_te.append(mae(yte, yte_hat))

            rmse_tr.append(rmse(ytr, ytr_hat))
            rmse_te.append(rmse(yte, yte_hat))

        # 結果保持
        self.results = {
            "mae_train": mae_tr,
            "mae_test": mae_te,
            "rmse_train": rmse_tr,
            "rmse_test": rmse_te
        }

        return self.results
