import numpy as np
import matplotlib.pyplot as plt
import typing
from Task3_HelperFunctions import computeRMSE


def kernelFunction(xi, xj, sigma):
    if isinstance(xi, float):
        return np.exp(-1/sigma**2 * (xi - xj)**2)
    return np.exp(-1/sigma**2 * np.linalg.norm(xi, xj)**2)


def main():
    # The settings for the "normal" plots
    kernelVariance = .15

    dataTrain = np.loadtxt("data_ml//training_data.txt")
    dataVal = np.loadtxt("data_ml//validation_data.txt")
    nTrain = len(dataTrain[0])
    nVal = len(dataVal[0])

    K = np.zeros((nTrain, nTrain))  # C-Matrix, see Bishop
    for i, xi in enumerate(dataTrain[0]):
        for j, xj in enumerate(dataTrain[0]):
            K[i, j] = kernelFunction(xi, xj, sigma=kernelVariance)

    K_inv = np.linalg.inv(K)

    xPlot = np.arange(0, 6.001, 0.01)
    yPlot = np.zeros(len(xPlot))
    for i, xi in enumerate(xPlot):
        k = np.zeros(nTrain)
        for j, xj in enumerate(dataTrain[0]):
            k[j] = kernelFunction(xi, xj, kernelVariance)
        yPlot[i] = k @ K_inv @ dataTrain[1]

    fig0, ax0 = plt.subplots()
    ax0.plot(xPlot, yPlot)
    plt.show()

    yPred = np.zeros(nVal)
    for i, xi in enumerate(dataVal[0]):
        k = np.zeros(nTrain)
        for j, xj in enumerate(dataTrain[0]):
            k[j] = kernelFunction(xi, xj, kernelVariance)
        yPred[i] = k @ K_inv @ dataTrain[1]

    RMSE = computeRMSE(yPred, dataVal[1])
    print(RMSE)
    # 0.24224710462885124

if __name__ == '__main__':
    main()
