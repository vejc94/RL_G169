# DMP-based controller.
#
# DMPPARAMS is the struct containing all the parameters of the DMP.
#
# PSI is the vector of basis functions.
#
# Q and QD are the current position and velocity, respectively.

import numpy as np

def dmpCtl (dmpParams, psi_i, q, qd):
    
    a = dmpParams.alpha*(dmpParams.beta*(dmpParams.goal - q))
    b = dmpParams.alpha*(qd/dmpParams.tau)
    fw = psi_i@dmpParams.w
    qdd = dmpParams.tau**2*(a - b + fw) 

    return qdd

