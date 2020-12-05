import numpy as np

A_t = np.array([[1, 0.1],
                [0, 1]])
B_t = np.array([0, 0.1])
b_t = np.array([5, 0])
cov_t = np.array([[0.01, 0],
                  [0, 0.01]])
K_t = np.array([5, 0.3])
k_t = 0.3
H_t = 1
R_t1 = np.array([[100000, 0],  # t = 14 or 40
                 [0, 0.1]])
R_t0 = np.array([[0.01, 0],  # else
                 [0, 0.1]])
r_t0 = np.array([10, 0])  # t <= 14
r_t1 = np.array([20, 0])  # else

T = 50
t1 = 14
t2 = 40
n = 20
