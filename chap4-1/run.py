import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from loader import DataLoader
from preprocessing import StandardScaler1D
from polynomial_model import PolynomialRegressor
from plotter import Plotter
from evaluator import mae, rmse
from experiment import PolynomialExperiment
from regularization import LassoPolynomialRegressor, lasso_path


def main():
    loader = DataLoader(save_path = "data/cars.csv", url = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/cars.csv")
    loader.data_load()
    speed, dist = loader.load()

    # 標準化
    scaler = StandardScaler1D()
    speed_std = scaler.fit_transform(speed)
    dist_std = scaler.fit_transform(dist)

    plotter = Plotter()
    xx = np.linspace(speed_std.min(), speed_std.max(), 200)

    for deg in range(1, 10):
        model = PolynomialRegressor(degree=deg)
        model.fit(speed_std, dist_std)
        yy = model.predict(xx)

        plotter.scatter_with_curve(
            speed_std, dist_std, xx, yy,
            degree=deg,
            save_path=f"output/poly_deg_{deg}.png"
        )

        # 評価
        experiment = PolynomialExperiment()
        results = experiment.run(speed_std, dist_std)
        plotter = Plotter()
        plotter.plot_metrics_2x2(
            results["mae_train"],
            results["mae_test"],
            results["rmse_train"],
            results["rmse_test"],
            save_prefix="output/")

    ols = PolynomialRegressor(degree=9)
    ols.fit(speed_std, dist_std)
    xx = np.linspace(speed_std.min(), speed_std.max(), 200)
    yy_ols = ols.predict(xx)

    # --- Lasso (例: alpha=0.1) ---
    lasso = LassoPolynomialRegressor(degree=9, alpha=0.1)
    lasso.fit(speed_std, dist_std)
    yy_lasso = lasso.predict(xx)

    # --- Plot OLS vs Lasso ---
    plotter = Plotter()
    plotter.plot_lasso_vs_ols(
        speed_std, dist_std, xx, yy_ols, yy_lasso,
        "output/ols_vs_lasso.png"
    )

    # --- 解パス ---
    coefs, log_lambdas, feature_names = lasso_path(speed_std, dist_std)
    plotter.plot_lasso_path(
        log_lambdas, coefs, feature_names,
        "output/lasso_path.png"
    )



if __name__ == "__main__":
    main()
