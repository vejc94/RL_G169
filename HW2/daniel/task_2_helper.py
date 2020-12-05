import matplotlib.pyplot as plt
from daniel.task_2_data import *

np.random.seed(0)


def get_s_tplus1(s_t, a_t) -> np.ndarray:
    return A_t.dot(s_t) + B_t.dot(a_t) + np.random.normal(b_t, np.sqrt(np.diagonal(cov_t)))


def get_mean(hist: list) -> np.ndarray:
    mean = np.sum(hist[i] for i in range(n)) / n
    return mean


def get_variance(hist: list) -> np.ndarray:
    mean = get_mean(hist)
    var = np.sum(hist[i] ** 2 for i in range(n)) / n - mean ** 2
    return var


def get_R_t(t):
    assert 0 <= t <= T
    if t == t1 or t == t2:
        return R_t1
    else:
        return R_t0


def get_r_t(t):
    assert 0 <= t <= T
    if t <= t1:
        return r_t0
    elif t1 < t <= T:
        return r_t1


def get_reward(states: np.ndarray, actions: np.ndarray) -> float:
    reward = 0.
    assert states.shape[0] == actions.shape[0]

    for i in range(states.shape[0] - 1):
        R_t = get_R_t(i)
        r_t = get_r_t(i)

        reward += -(states[i, :] - r_t) @ R_t @ (states[i, :] - r_t).T - actions[i].T * H_t * actions[i]

    reward += -(states[-1, :] - r_t1) @ R_t0 @ (states[-1, :] - r_t1).T
    return reward


def plotStuff(mean_list: list, var_list: list, labels_list: list, title: str):
    multiples_std = 1.96
    fig0, ax0 = plt.subplots(1)
    i = 0
    for mean, var in zip(mean_list, var_list):
        ax0.fill_between(np.arange(0, T + 1),
                         mean[:] + multiples_std * np.sqrt(var[:]) / np.sqrt(n),
                         mean[:] - multiples_std * np.sqrt(var[:]) / np.sqrt(n), alpha=.5)
        ax0.plot(mean[:], label=labels_list[i])
        i += 1

    ax0.set_title(title)
    ax0.set_xlim(0, T)
    ax0.grid()
    ax0.legend()
    plt.tight_layout()
    return fig0
