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
        Phi[k, :] = (Phi[k, :] * time_k) / np.sum(Phi[k, :])  # Normalize basis functions and weight by canonical state

    return Phi
