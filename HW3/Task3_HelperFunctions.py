import numpy as np


def featureFunction(x, i):
    return np.sin(2**i * x)


def computeRMSE(yPred: np.ndarray, y: np.ndarray) -> float:
    """Returns the RMSE between the target calculated with input x and weights and the true value of y. The used
    formulas are shown as comments in the code"""
    RMSE = 0
    for i, yi in enumerate(y):
        # RMSE += (f(x_i) - y_i)**2
        RMSE += (yPred[i] - yi) ** 2
    # RMSE = RMSE/N  Normalizes by the number of samples
    RMSE /= len(y)
    # RMSE = np.sqrt(RMSE)  Takes the square root
    RMSE = RMSE ** .5
    return RMSE