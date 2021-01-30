import numpy as np
import matplotlib.pyplot as plt


def getProMPBasis(dt, nSteps, n_of_basis, bandwidth):
    nBasis = n_of_basis
    time = np.arange(dt, nSteps * dt, dt)
    assert nSteps == len(time)
    Ts = nSteps * dt - dt

    C = np.zeros(nBasis)  # Basis function centres
    H = np.zeros(nBasis)  # Basis function bandwidths

    for i in range(nBasis):
        C[i] = -2 * bandwidth + (Ts + 4 * bandwidth) / nBasis * i

    for i in range(nBasis):
        H[i] = bandwidth ** (1 / 4)

    Phi = np.zeros((nSteps, nBasis))

    for k, time_k in enumerate(time):
        for j in range(nBasis):
            Phi[k, j] = np.exp(-.5 * (time_k - C[j]) ** 2 / H[j])  # Basis function activation over time
    for k in range(Phi.shape[0]):
        Phi[k, :] = (Phi[k, :]) / np.sum(Phi[k, :])  # Normalize basis functions and weight by canonical state

    return Phi


if __name__ == '__main__':
    dt = 0.002
    time = np.arange(dt, 3, dt)
    nSteps = len(time)
    nBasis = 30
    bandwidth = 0.2
    Phi = getProMPBasis(dt, nSteps, nBasis, bandwidth)

    fig0, ax0 = plt.subplots(1)
    for i in range(Phi.shape[1]):
        ax0.plot(time, Phi[:,i])
        #ax0.plot(time, summPhi[:,i] for i in range(Phi.shape[1])))

    plt.show()