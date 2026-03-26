from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def mae(true, pred):
    return mean_absolute_error(true, pred)

def rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))