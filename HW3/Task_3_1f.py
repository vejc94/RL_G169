import numpy as np
import matplotlib.pyplot as plt
from Task3_HelperFunctions import featureFunction


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


def main():
    dataTrain = np.loadtxt("data_ml//training_data.txt")

    nList = np.arange(1, 10)
    RMSEMeanList = np.zeros(len(nList))
    RMSEVarList = np.zeros(len(nList))

    for i, n in enumerate(nList):
        RMSEList = np.zeros(len(dataTrain[0]))
        for j, xj in enumerate(dataTrain[0]):
            dataTrain_kFold = dataTrain[:, np.arange(len(dataTrain[0])) != j]
            phi = np.zeros((len(dataTrain_kFold[0]), n))
            for k in range(n):
                phi[:, k] = featureFunction(dataTrain_kFold[0], k)

            w_hat = np.linalg.inv(phi.T @ phi) @ phi.T @ dataTrain_kFold[1]

            phiVal = np.zeros((1, n))
            for k in range(n):
                phiVal[:, k] = featureFunction(xj, k)
            yPred = phiVal.dot(w_hat)
            RMSEList[j] = ((yPred - dataTrain[1, j])**2)**.5

        RMSEMeanList[i] = np.mean(RMSEList)
        RMSEVarList[i] = np.var(RMSEList)

    fig0, ax0 = plt.subplots()
    ax0.set_title('Mean RMSE of the Training Set using kFold-Cross-Validation')
    ax0.plot(nList, RMSEMeanList, color='blue', label='mean')
    ax1 = ax0.twinx()
    ax1.plot(nList, RMSEVarList, color='red', label='variance')
    ax0.set_xlabel('Number of Features')
    ax0.set_ylabel('RMSE')
    ax0.set_xlim((1, 9))
    ax0.legend()
    ax1.legend(loc='upper center')
    plt.savefig('plots/Task_3_1f')


if __name__ == '__main__':
    main()
