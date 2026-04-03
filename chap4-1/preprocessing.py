import numpy as np

class StandardScaler1D:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x: np.ndarray):
        self.mean_ = np.mean(x)
        self.std_ = np.std(x)

    def transform(self, x: np.ndarray):
        return (x - self.mean_) / self.std_

    def fit_transform(self, x: np.ndarray):
        self.fit(x)
        return self.transform(x)