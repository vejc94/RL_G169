import numpy as np
import matplotlib.pyplot as plt
from Task3_HelperFunctions import featureFunction


def main():
    dataTrain = np.loadtxt("data_ml//training_data.txt")
    dataVal = np.loadtxt("data_ml//validation_data.txt")

    nSamples = dataTrain.shape[1]
    nList = [2, 3, 9]
    xRange = np.linspace(0, 6, 100)

    fig0, ax0 = plt.subplots()
    ax0.scatter(dataTrain[0], dataTrain[1], label='Training')
    ax0.scatter(dataVal[0], dataVal[1], label='Validation')

    for i, n in enumerate(nList):
        phi = np.zeros((nSamples, n))
        for j in range(n):
            phi[:, j] = featureFunction(dataTrain[0], j)

        # Get the weights
        w_hat = np.linalg.inv(phi.T @ phi) @ phi.T @ dataTrain[1]

        phiPred = np.zeros((len(xRange), n))
        for j in range(n):
            phiPred[:, j] = featureFunction(xRange, j)
        yPred = phiPred.dot(w_hat)

        ax0.plot(xRange, yPred, label=f"Degree {n}")

    ax0.legend()
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    ax0.set_xlim((0, 6))
    ax0.grid()
    plt.savefig('plots/Task_3_1c')


if __name__ == '__main__':
    main()
