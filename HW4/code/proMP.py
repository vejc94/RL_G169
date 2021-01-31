import numpy as np
import matplotlib.pyplot as plt
from getImitationData import *
from getProMPBasis import *

def proMP (nBasis, condition=False):

    dt = 0.002
    time = np.arange(dt,3,dt)
    nSteps = len(time)
    data = getImitationData(dt, time, multiple_demos=True)
    q = data[0]
    qd = data[1]
    qdd = data[2]
    bandwidth = 0.2
    
    Phi = getProMPBasis( dt, nSteps, nBasis, bandwidth )
    
    pseudo_inv = np.linalg.inv(Phi.T@Phi)
    w = pseudo_inv@Phi.T@q.T
    mean_w = np.mean(w, axis=1)
    cov_w = np.cov(w, rowvar=True)
    mean_traj = Phi@mean_w
    
    std_traj = np.sqrt(np.diag(0.01*np.eye(nSteps) + Phi@cov_w@Phi.T))

    plt.figure()
    plt.fill_between(time, mean_traj - 2*std_traj, mean_traj + 2*std_traj, alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')
    plt.plot(time, mean_traj, color='#1B2ACC')
    plt.plot(time, q.T)
    plt.title('ProMP with ' + str(nBasis) + ' basis functions')

    #Conditioning
    if condition:
        y_d = 3
        Sig_d = 0.0002
        t_point = np.round(2300/2).astype(np.int)

        tmp = np.dot(cov_w, Phi[t_point,:]) / (Sig_d + np.dot(Phi[t_point,:].T,np.dot(cov_w,Phi[t_point,:])))

        cov_w_new = cov_w - tmp@Phi[t_point,:].T * cov_w
        mean_w_new = mean_w + tmp*(y_d - Phi[t_point,:].T@mean_w)
        
        traj_new = np.zeros((10, 1499))
        for i in range(10):
            w_new = np.random.normal(mean_w_new, np.std(np.diagonal(cov_w_new)))
            traj_new[i] = Phi@w_new
        mean_traj_new = np.mean(traj_new, axis=0)
        std_traj_new = np.std(traj_new, axis=0)

        plt.figure()
        plt.fill_between(time, mean_traj - 2*std_traj, mean_traj + 2*std_traj, alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')
        plt.plot(time,mean_traj, color='#1B2ACC')
        plt.fill_between(time, mean_traj_new - 2*std_traj_new, mean_traj_new + 2*std_traj_new, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        plt.plot(time, mean_traj_new, color='#CC4F1B')

        sample_traj = np.dot(Phi, np.random.multivariate_normal(mean_w_new,cov_w_new,10).T)
        plt.plot(time, sample_traj)
        plt.title('ProMP after contidioning with new sampled trajectories')

    plt.draw_all()
    plt.pause(0.001)
