import numpy as np
import matplotlib.pyplot as plt
from daniel.task_2_data import *

V_t = np.zeros((T+1, 2, 2))
v_t = np.zeros((T+1, 2))

V_t[-1,:,:] = R_t0
v_t[-1,:] = r_t1

for t in np.arange(T-1, 0, -1):
    if t==t1 or t==t2:
        R_t = R_t1
    else:
        R_t = R_t0
    if t<=t1:
        r_t = r_t0
    else:
        r_t = r_t1

    M_t = B_t.dot((H_t + B_t.T @ V_t[t+1,:,:] @ B_t)**-1) @ (B_t.T @ V_t[t+1,:,:] @ A_t)
    V_t[t,:,:] = R_t + (A_t - M_t).T @ V_t[t+1,:,:] @ A_t
    v_t[t,:] = R_t@r_t + (A_t - M_t).T @ (v_t[t+1,:] - V_t[t+1,:,:] @ b_t)

states = np.zeros((T+1, 2))
states[0,:] = np.random.normal((0, 0), (1, 1), 2)

actions = np.zeros((T+1))

for t in np.arange(0, T, 1):
    actions[t] = - (H_t + B_t.T @ V_t[t+1,:,:] @ B_t)**-1 * (B_t @ (V_t[t+1,:,:] @ (A_t @ states[t,:] + b_t) - v_t[t+1,:]))
    states[t+1] = A_t @ states[t,:] + B_t.dot(actions[t]) + np.random.normal(b_t, np.diagonal(cov_t))

fig0, ax = plt.subplots(3)
ax[0].plot(states[:,0])
ax[1].plot(states[:,1])
ax[2].plot(actions)
plt.show()
