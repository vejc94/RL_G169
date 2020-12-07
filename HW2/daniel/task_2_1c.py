from daniel.task_2_helper import *


def get_a_t(s_t, V_tplus1, v_tplus1, t) -> float:
    assert 0 <= t <= T-1
    return -(H_t + B_t.T.dot(V_tplus1.dot(B_t)))**-1 * (B_t.T.dot(V_tplus1.dot(A_t.dot(s_t) + b_t) - v_tplus1))


def execute_LQR() -> (np.ndarray, np.ndarray):
    s_0 = np.random.normal((0, 0), np.sqrt((1, 1)))

    s_t_array = np.zeros((T + 1, 2))
    s_t_array[0, :] = s_0

    a_t_array = np.zeros((T+1))

    V_t_array = [None] * (T+1)
    v_t_array = [None] * (T+1)

    for t in np.arange(T, 0, -1):
        V_t_array[t] = get_V_t(V_t_array, t)
        v_t_array[t] = get_v_t(V_t_array, v_t_array, t)

    a_t_array[0] = get_a_t(s_t_array[0, :], V_tplus1=V_t_array[1], v_tplus1=v_t_array[1], t=0)
    for t in np.arange(1, T, 1):
        s_t_array[t, :] = get_s_tplus1(s_t_array[t - 1, :], a_t_array[t - 1])
        a_t_array[t] = get_a_t(s_t_array[t, :], V_tplus1=V_t_array[t+1], v_tplus1=v_t_array[t+1], t=t)
    s_t_array[T, :] = get_s_tplus1(s_t_array[T - 1, :], a_t_array[T - 1])

    return s_t_array, a_t_array


def get_V_t(V_t_array, t):
    assert 1 <= t <= T
    if t == T:
        return get_R_t(t)
    elif 1 <= t <= T-1:
        return get_R_t(t) + (A_t - get_M_t(V_t_array[t+1], t)).T.dot(V_t_array[t+1].dot(A_t))


def get_v_t(V_t_array, v_t_array, t):
    assert 1 <= t <= T
    if t == T:
        return get_R_t(t).dot(get_r_t(t))
    elif 1 <= t <= T-1:
        return get_R_t(t).dot(get_r_t(t)) \
               + (A_t - get_M_t(V_t_array[t+1], t)).T.dot(v_t_array[t+1] - V_t_array[t+1].dot(b_t))


def get_M_t(V_tplus1, t):
    assert 0 <= t <= T-1
    return B_t.reshape([2,1]).dot((((H_t + B_t.T @ V_tplus1 @ B_t) ** -1) * (B_t.T @ V_tplus1 @ A_t)).reshape([1,2]))


def task_2_1c():
    s_t_history = list()
    a_t_history = list()
    rew_history = list()
    for i in range(n):
        s_t_temp, a_t_temp = execute_LQR()
        # fig1, ax = plt.subplots(3)
        # ax[0].plot(s_t_temp[:,0])
        # ax[1].plot(s_t_temp[:,1])
        # ax[2].plot(a_t_temp)
        # plt.show()
        s_t_history.append(s_t_temp)
        a_t_history.append(a_t_temp)

        rew_history.append(get_reward(s_t_temp, a_t_temp))

    s_t_mean, a_t_mean = get_mean(s_t_history), get_mean(a_t_history)
    s_t_var, a_t_var = get_variance(s_t_history), get_variance(a_t_history)
    rew_mean, rew_var = get_mean(rew_history), get_variance(rew_history)

    print(f"Task 2.1c Reward Mean: {rew_mean}, Reward Std: {np.sqrt(rew_var)}")

    return s_t_mean, a_t_mean, s_t_var, a_t_var


if __name__ == "__main__":
    from daniel.task_2_1a import task_2_1a
    from daniel.task_2_1b import task_2_1b
    np.random.seed(0)

    s_t_mean_21a, a_t_mean_21a, s_t_var_21a, a_t_var_21a = task_2_1a()
    s_t_mean_21b, a_t_mean_21b, s_t_var_21b, a_t_var_21b = task_2_1b()
    s_t_mean_21c, a_t_mean_21c, s_t_var_21c, a_t_var_21c = task_2_1c()

    labels = [["Task 2.1a", "Task 2.1b", "Task 2.1c"]]*3
    titles = ["State 1", "State 2", "Control"]
    ii = 0
    for mean, var in zip(([s_t_mean_21a[:,0], s_t_mean_21b[:,0], s_t_mean_21c[:,0]],
                          [s_t_mean_21a[:,1], s_t_mean_21b[:,1], s_t_mean_21c[:,1]],
                          [a_t_mean_21a, a_t_mean_21b, a_t_mean_21c]),
                         ([s_t_var_21a[:,0], s_t_var_21b[:,0], s_t_var_21c[:,0]],
                          [s_t_var_21a[:,1], s_t_var_21b[:,1], s_t_var_21c[:,1]],
                          [a_t_var_21a, a_t_var_21b, a_t_var_21c])):
        fig = plotStuff(mean, var, labels[ii], titles[ii])
        fig.savefig("daniel/plots/Task2_1c_" + titles[ii] + ".pdf")
        ii += 1
        plt.show()
