from daniel.task_2_helper import *


def get_a_t(s_t) -> np.ndarray:
    return -K_t.dot(s_t) + k_t


def execute_LQR() -> (np.ndarray, np.ndarray):
    s_0 = np.random.normal((0, 0), np.sqrt((1, 1)))

    s_t_array = np.zeros((T + 1, 2))
    s_t_array[0, :] = s_0

    a_t_array = np.zeros((T + 1))
    a_t_array[0] = get_a_t(s_t_array[0, :])

    for i in np.arange(1, T + .1, 1, dtype=int):
        s_t_array[i, :] = get_s_tplus1(s_t_array[i - 1, :], a_t_array[i - 1])
        a_t_array[i] = get_a_t(s_t_array[i, :])

    return s_t_array, a_t_array


def task_2_1a():
    s_t_history = list()
    a_t_history = list()
    rew_history = list()
    for i in range(n):
        s_t_temp, a_t_temp = execute_LQR()
        s_t_history.append(s_t_temp)
        a_t_history.append(a_t_temp)

        rew_history.append(get_reward(s_t_temp, a_t_temp))

    s_t_mean, a_t_mean = get_mean(s_t_history), get_mean(a_t_history)
    s_t_var, a_t_var = get_variance(s_t_history), get_variance(a_t_history)
    rew_mean, rew_var = get_mean(rew_history), get_variance(rew_history)

    print(f"Task 2.1a Reward Mean: {rew_mean}, Reward Std: {np.sqrt(rew_var)}")

    return s_t_mean, a_t_mean, s_t_var, a_t_var


if __name__ == "__main__":
    np.random.seed(0)
    s_t_mean_21a, a_t_mean_21a, s_t_var_21a, a_t_var_21a = task_2_1a()

    labels = [["Task 2.1a"]]*3
    titles = ["State 1", "State 2", "Control"]
    ii = 0
    for mean, var in zip(([s_t_mean_21a[:,0]], [s_t_mean_21a[:,1]], [a_t_mean_21a]),
                         ([s_t_var_21a[:,0]], [s_t_var_21a[:,1]], [a_t_var_21a])):
        plotStuff(mean, var, labels[ii], titles[ii])
        ii += 1
        plt.show()
