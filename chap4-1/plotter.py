import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def scatter_with_curve(self, x, y, x_line, y_line, degree, save_path):
        plt.figure()
        plt.scatter(x, y)
        plt.plot(x_line, y_line, color="blue")
        plt.title(f"Polynomial Regression (degree={degree})")
        plt.xlabel("Speed")
        plt.ylabel("Distance")
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def plot_metric(self, train, test, title, savepath):
        degrees = range(1, len(train)+1)

        plt.figure()
        plt.plot(degrees, train, label="Train")
        plt.plot(degrees, test, label="Test")
        plt.title(title)
        plt.xlabel("Degree")
        plt.ylabel(title)
        plt.grid(True)
        plt.legend()
        plt.savefig(savepath)
        plt.close()

    def plot_metric_single(self, values, title, ylabel, savepath):
        degrees = range(1, len(values) + 1)

        plt.figure()
        plt.plot(degrees, values, marker="o")
        plt.title(title)
        plt.xlabel("Degree")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.savefig(savepath)
        plt.close()

    def plot_metrics_2x2(self, mae_tr, mae_te, rmse_tr, rmse_te, save_prefix="output/"):
        """MAE/RMSE × Train/Test の2×2図をまとめて保存"""

        # 図1: train MAE
        self.plot_metric_single(
            mae_tr,
            title="Train MAE",
            ylabel="MAE",
            savepath=f"{save_prefix}train_mae.png"
        )

        # 図2: test MAE
        self.plot_metric_single(
            mae_te,
            title="Test MAE",
            ylabel="MAE",
            savepath=f"{save_prefix}test_mae.png"
        )

        # 図3: train RMSE
        self.plot_metric_single(
            rmse_tr,
            title="Train RMSE",
            ylabel="RMSE",
            savepath=f"{save_prefix}train_rmse.png"
        )

        # 図4: test RMSE
        self.plot_metric_single(
            rmse_te,
            title="Test RMSE",
            ylabel="RMSE",
            savepath=f"{save_prefix}test_rmse.png"
        )


    def plot_lasso_vs_ols(self, x, y, xx, yy_ols, yy_lasso, savepath):
        plt.figure()
        plt.scatter(x, y, alpha=0.6)
        plt.plot(xx, yy_ols, label="OLS (deg=9)", color="blue")
        plt.plot(xx, yy_lasso, label="Lasso", color="red")
        plt.title("9-degree OLS vs Lasso Regression")
        plt.xlabel("Speed (standardized)")
        plt.ylabel("Distance")
        plt.grid(True)
        plt.legend()
        plt.savefig(savepath)
        plt.close()

    def plot_lasso_path(self, log_lambdas, coefs, feature_names, savepath):
        plt.figure()

        for i in range(1, len(feature_names)):  # bias は除外
            plt.plot(log_lambdas, coefs[:, i], label=feature_names[i])

        plt.title("Lasso Solution Path")
        plt.xlabel("log(lambda)")
        plt.ylabel("Coefficient")
        plt.grid(True)
        plt.legend()
        plt.savefig(savepath)
        plt.close()

