import numpy as np

#Mean squared Error

# E = 1/n sigma(Y*-Y)^2
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

# dE/dY = 2/n *(Y-Y*)
def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

