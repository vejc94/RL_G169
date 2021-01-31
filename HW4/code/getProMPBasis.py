import numpy as np
import matplotlib.pyplot as plt

def getProMPBasis(dt, nSteps, n_of_basis, bandwidth):
    
    time = np.arange(dt,nSteps*dt,dt)
    tau = 1
    alpha_z = 3/time[-1]
    low = -2*bandwidth
    high = time[-1] + 2*bandwidth
    
    C = np.zeros(n_of_basis)  # Basis function centres
    H = np.zeros(n_of_basis)  # Basis function bandwidths
    
    C= np.linspace(-2*bandwidth, time[-1] + 2*bandwidth, n_of_basis)

    for i in range(n_of_basis):
        H[i] = bandwidth ** (1 / 4)
    H[:] = bandwidth
    
    z = np.exp(-alpha_z*tau*time)
    
    Phi = np.zeros((nSteps, n_of_basis))    
    
    for k in range(nSteps):
        for j in range(n_of_basis):
            Phi[k,j] = np.exp(-1/2*( time[k] - C[j] )**2/H[j] ) # Basis function activation over time
    
    for k in range(nSteps):    
        Phi[k,:] = (Phi[k,:]) / np.sum( Phi[k,:]) # Normalize basis functions and weight by canonical state
        
    plot_proMP(Phi, n_of_basis, time)
    return Phi

def plot_proMP(Phi, n_of_basis, time):
    for i in range(n_of_basis):
        plt.plot(time, Phi[:, i])

        
    
    
