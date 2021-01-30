import numpy as np
import dmpComparison

# Learns the weights for the basis functions.
#
# Q_IM, QD_IM, QDD_IM are vectors containing positions, velocities and
# accelerations of the two joints obtained from the trajectory that we want
# to imitate.
#
# DT is the time step.
#
# NSTEPS are the total number of steps.

from getDMPBasis import *
import numpy as np


class dmpParams():
    def __init__(self):
        self.alphaz = 0.0
        self.alpha = 0.0
        self.beta = 0.0
        self.Ts = 0.0
        self.tau = 0.0
        self.nBasis = 0.0
        self.goal = 0.0
        self.w = 0.0


def dmpTrain(q, qd, qdd, dt, nSteps):
    params = dmpParams()
    # Set dynamic system parameters
    params.alphaz = 3 / (nSteps * dt)
    params.alpha = 25
    params.beta = 6.25
    params.Ts = nSteps * dt
    params.tau = 1
    params.nBasis = 50
    params.goal = q[:, -1]

    Phi = getDMPBasis(params, dt, nSteps)

    # Compute the forcing function
    a = qdd / params.tau ** 2
    b = params.alpha * params.beta * (params.goal[:, np.newaxis] - q)
    c = params.alpha * qd / params.tau
    ft = a - b + c

    # Learn the weights
    sigma = 0.1
    N, M = Phi.shape
    pseudo_inv = np.linalg.inv(Phi.T @ Phi + sigma ** 2 * np.eye(M, M))
    params.w = pseudo_inv @ Phi.T @ ft.T
