import numpy as np
import matplotlib.pyplot as plt
from Task3_HelperFunctions import featureFunction, computeRMSE


def main():
    dataTrain = np.loadtxt("data_ml//training_data.txt")
    dataVal = np.loadtxt("data_ml//validation_data.txt")

    nSamples = dataTrain.shape[1]
    nList = np.arange(1, 10)
    RMSEList = np.zeros(len(nList))
    RMSEValList = np.zeros(len(nList))

    for i, n in enumerate(nList):

        # Task 1d and 1e
        phi = np.zeros((nSamples, n))
        for j in range(n):
            phi[:, j] = featureFunction(dataTrain[0], j)

        w_hat = np.linalg.inv(phi.T @ phi) @ phi.T @ dataTrain[1]

        yPred = phi.dot(w_hat)
        RMSEList[i] = computeRMSE(yPred, dataTrain[1])

        # Task 1e
        phiVal = np.zeros((dataVal.shape[1], n))
        for j in range(n):
            phiVal[:, j] = featureFunction(dataVal[0], j)
        yPredVal = phiVal.dot(w_hat)
        RMSEValList[i] = computeRMSE(yPredVal, dataVal[1])

    fig0, ax0 = plt.subplots()
    ax0.set_title('RMSE of the Training Set')
    ax0.plot(nList, RMSEList)
    ax0.set_xlabel('Number of Features')
    ax0.set_ylabel('RMSE')
    ax0.set_xlim((1, 9))
    ax0.grid()
    plt.savefig('plots/Task_3_1d')

    fig1, ax1 = plt.subplots()
    ax1.set_title('RMSE of the Training and Validation Set')
    ax1.plot(nList, RMSEList, label="Training")
    ax1.plot(nList, RMSEValList, label="Validation")
    ax1.legend()
    ax1.set_xlabel('Number of Features')
    ax1.set_ylabel('RMSE')
    ax1.set_xlim((1, 9))
    ax1.grid()
    plt.savefig('plots/Task_3_1e')


if __name__ == '__main__':
    main()
