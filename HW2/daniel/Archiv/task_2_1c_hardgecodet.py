import numpy as np
import matplotlib.pyplot as plt


def get_reward(states: np.ndarray, actions: np.ndarray) -> float:
    reward = 0.
    assert states.shape[0] == actions.shape[0]

    for i in range(states.shape[0] - 1):
        if i == 14 or i == 40:
            R_t = np.array([[100000, 0], [0, 0.1]])
        else:
            R_t = np.array([[0.01, 0], [0, 0.1]])
        if i <= 14:
            r_t = np.array([10, 0])
        else:
            r_t = np.array([20, 0])

        reward += -(states[i, :] - r_t.T) @ R_t @ (states[i, :] - r_t.T).T - actions[i].T * H_t * actions[i]

    reward += -(states[-1, :] - np.array([20, 0]).T) @ np.array([[0.01, 0], [0, 0.1]]) @ (states[-1, :] - np.array([20, 0]).T).T
    return reward


A_t = np.array([[1, 0.1],
                [0, 1]])
B_t = np.array([0, 0.1])
b_t = np.array([5, 0])
cov_t = np.array([[0.01, 0],
                  [0, 0.01]])

H_t = 1
T = 50

V_t = np.zeros((T + 1, 2, 2))
v_t = np.zeros((T + 1, 2))

V_t[-1, :, :] = np.array([[0.01, 0], [0, 0.1]])
v_t[-1, :] = np.array([20, 0])

for t in np.arange(T - 1, 0, -1):
    if t == 14 or t == 40:
        R_t = np.array([[100000, 0], [0, 0.1]])
    else:
        R_t = np.array([[0.01, 0], [0, 0.1]])
    if t <= 14:
        r_t = np.array([10, 0])
    else:
        r_t = np.array([20, 0])

    M_t = B_t.reshape([2,1]).dot((((H_t + B_t.T @ V_t[t + 1, :, :] @ B_t) ** -1) * (B_t.T @ V_t[t + 1, :, :] @ A_t)).reshape([1,2]))
    V_t[t, :, :] = R_t + (A_t - M_t).T @ V_t[t + 1, :, :] @ A_t
    v_t[t, :] = R_t @ r_t + (A_t - M_t).T @ (v_t[t + 1, :] - V_t[t + 1, :, :] @ b_t)

states = np.zeros((T + 1, 2))
states[0, :] = np.random.normal((0, 0), (1, 1), 2)

actions = np.zeros((T + 1))

for t in np.arange(0, T, 1):
    actions[t] = - (H_t + B_t.T @ V_t[t + 1, :, :] @ B_t) ** -1 * (
                B_t @ (V_t[t + 1, :, :] @ (A_t @ states[t, :] + b_t) - v_t[t + 1, :]))
    states[t + 1, :] = A_t @ states[t, :] + B_t.dot(actions[t]) + np.random.normal(b_t, np.diagonal(cov_t))

fig0, ax = plt.subplots(3)
ax[0].plot(states[:, 0])
ax[1].plot(states[:, 1])
ax[2].plot(actions)
plt.show()


print(get_reward(states, actions))