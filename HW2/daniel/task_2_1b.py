from daniel.task_2_helper import *


def get_a_t(s_t, s_t_des) -> np.ndarray:
    return K_t @ (s_t_des - s_t) + k_t


def execute_LQR(useTarget) -> (np.ndarray, np.ndarray):
    s_0 = np.random.normal((0, 0), (1, 1), 2)
    if useTarget:
        s_t_des = r_t0[:,0].T
    else:
        s_t_des = np.zeros(2)

    s_t_array = np.zeros((T + 1, 2))
    s_t_array[0, :] = s_0

    a_t_array = np.zeros((T + 1))
    a_t_array[0] = get_a_t(s_t_array[0, :], s_t_des)

    for i in np.arange(1, T + 1, 1):
        if useTarget: s_t_des = get_r_t(i)[:,0]
        s_t_array[i, :] = get_s_tplus1(s_t_array[i - 1, :], a_t_array[i - 1])
        a_t_array[i] = get_a_t(s_t_array[i, :], s_t_des)

    return s_t_array, a_t_array


def task_2_1b():
    s_t_history = list()
    a_t_history = list()
    rew_history = list()
    for i in range(n):
        s_t_temp, a_t_temp = execute_LQR(useTarget=True)
        s_t_history.append(s_t_temp)
        a_t_history.append(a_t_temp)
        rew_history.append(get_reward(s_t_temp, a_t_temp))
    s_t_mean, a_t_mean = get_mean(s_t_history), get_mean(a_t_history)
    s_t_var, a_t_var = get_variance(s_t_history), get_variance(a_t_history)
    rew_mean, rew_var = get_mean(rew_history), get_variance(rew_history)

    print(f"Task 2.1b Reward Mean: {rew_mean}, Reward Std: {np.sqrt(rew_var)}")

    return s_t_mean, a_t_mean, s_t_var, a_t_var


if __name__ == "__main__":
    from daniel.task_2_1a import task_2_1a
    np.random.seed(0)

    s_t_mean_21a, a_t_mean_21a, s_t_var_21a, a_t_var_21a = task_2_1a()
    s_t_mean_21b, a_t_mean_21b, s_t_var_21b, a_t_var_21b = task_2_1b()

    labels = [["Task 2.1a", "Task 2.1b"]]*3
    titles = ["State 1", "State 2", "Control"]
    ii = 0
    for mean, var in zip(([s_t_mean_21a[:,0], s_t_mean_21b[:,0]],
                          [s_t_mean_21a[:,1], s_t_mean_21b[:,1]],
                          [a_t_mean_21a, a_t_mean_21b]),
                         ([s_t_var_21a[:,0], s_t_var_21b[:,0]],
                          [s_t_var_21a[:,1], s_t_var_21b[:,1]],
                          [a_t_var_21a, a_t_var_21b])):
        plotStuff(mean, var, labels[ii], titles[ii])
        ii += 1
        plt.show()