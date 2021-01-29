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
    params.alphaz = 3 / (nSteps * dt)  # ToDo: Not sure about this one
    params.alpha = 25
    params.beta = 6.25
    params.Ts = nSteps * dt  # ToDo: Not sure about this one
    params.tau = 1
    params.nBasis = 50
    params.goal = np.asarray(q[:, -1]) # np.asarray([0.3, -0.8])

    # Daniel: This should actually be Psi, right?
    Phi = getDMPBasis(params, dt, nSteps)

    # Compute the forcing function
    ft = 1 / params.tau ** 2 * (0 - qdd) + 1 / params.tau * (0 - qd) \
         + params.alpha * params.beta * (params.goal.reshape(2, 1)  - q)  #

    # Learn the weights
    sigma = 10 ** -12
    params.w = np.linalg.inv(Phi.transpose().dot(Phi) + sigma ** 2 * np.identity(Phi.shape[1])).dot(Phi.transpose()) \
        .dot(ft.transpose())

    return params
