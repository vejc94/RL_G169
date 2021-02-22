import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


##
def genGridWorld():
    O = -1e5  # Dangerous places to avoid
    D = 35    # Dirt
    W = -100  # Water
    C = -3000 # Cat
    T = 1000  # Toy
    grid_list = {0:'', O:'O', D:'D', W:'W', C:'C', T:'T'}
    grid_world = np.array([[0, O, O, 0, 0, O, O, 0, 0, 0],
        [0, 0, 0, 0, D, O, 0, 0, D, 0],
        [0, D, 0, 0, 0, O, 0, 0, O, 0],
        [O, O, O, O, 0, O, 0, O, O, O],
        [D, 0, 0, D, 0, O, T, D, 0, 0],
        [0, O, D, D, 0, O, W, 0, 0, 0],
        [W, O, 0, O, 0, O, D, O, O, 0],
        [W, 0, 0, O, D, 0, 0, O, D, 0],
        [0, 0, 0, D, C, O, 0, 0, D, 0]])
    return grid_world, grid_list


##
def showWorld(grid_world, tlt):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title(tlt)
    ax.set_xticks(np.arange(0.5,10.5,1))
    ax.set_yticks(np.arange(0.5,9.5,1))
    ax.grid(color='b', linestyle='-', linewidth=1)
    ax.imshow(grid_world, interpolation='nearest', cmap='copper')
    return ax


##
def showTextState(grid_world, grid_list, ax):
    for x in range(grid_world.shape[0]):
        for y in range(grid_world.shape[1]):
            if grid_world[x,y] >= -3000:
                ax.annotate(grid_list.get(grid_world[x,y]), xy=(y,x), horizontalalignment='center')


##
def showPolicy(policy, ax):
    for x in range(policy.shape[0]):
        for y in range(policy.shape[1]):
            if policy[x,y] == 0:
                ax.annotate('$\downarrow$', xy=(y,x), horizontalalignment='center')
            elif policy[x,y] == 1:
                ax.annotate(r'$\rightarrow$', xy=(y,x), horizontalalignment='center')
            elif policy[x,y] == 2:
                ax.annotate(r'$\uparrow$', xy=(y,x), horizontalalignment='center')
            elif policy[x,y] == 3:
                ax.annotate('$\leftarrow$', xy=(y,x), horizontalalignment='center')
            elif policy[x,y] == 4:
                ax.annotate('$\perp$', xy=(y,x), horizontalalignment='center')


##
def ValIter(R, discount, maxSteps, infHor, probModel=None):
    T = maxSteps
    rows, columns = R.shape
    V = np.zeros([rows, columns, T])
    V[:, :, -1] = np.copy(R)
    Q = np.zeros([rows, columns, 5])
    
    if not infHor:
        for i in range(T-2, -1, -1):
            for k in range(rows):  # row
                for l in range(columns):  # column
                    for a in range(5):  # iterate over actions
                        if not probModel:
                            new_state = trans_model(np.array([k, l]), a)
                            Q[k, l, a] = R[k, l] + V[new_state[0], new_state[1], i+1]
                       
                        else:
                            if a == 0 or a == 2:  # up or down
                                new_st_1 = trans_model(np.array([k, l]), a)
                                new_st_2 = trans_model(np.array([k, l]), 1)
                                new_st_3 = trans_model(np.array([k, l]), 3)
                                new_st_4 = trans_model(np.array([k, l]), 4)
                                
                                Q[k, l, a] = R[k, l] + 0.7*V[new_st_1[0], new_st_1[1], i+1]\
                                            + 0.1*V[new_st_2[0], new_st_2[1], i+1]\
                                            + 0.1*V[new_st_3[0], new_st_3[1], i+1]\
                                            + 0.1*V[new_st_4[0], new_st_4[1], i+1]\
                            
                            elif a == 1 or a == 3:  # right or left
                                new_st_1 = trans_model(np.array([k, l]), a)
                                new_st_2 = trans_model(np.array([k, l]), 0)
                                new_st_3 = trans_model(np.array([k, l]), 2)
                                new_st_4 = trans_model(np.array([k, l]), 4)
                                
                                Q[k, l, a] = R[k, l] + 0.7*V[new_st_1[0], new_st_1[1], i+1]\
                                            + 0.1*V[new_st_2[0], new_st_2[1], i+1]\
                                            + 0.1*V[new_st_3[0], new_st_3[1], i+1]\
                                            + 0.1*V[new_st_4[0], new_st_4[1], i+1]
        
                            elif a == 4:  # stay
                                new_state = trans_model(np.array([k, l]), a)
                                Q[k, l, a] = R[k, l] + V[new_state[0], new_state[1], i+1]
                 
            V[:,:, i] = np.max(Q, axis=2)
    else:
        diff = np.inf
        V = np.copy(R)
        V_old = np.zeros([rows, columns])
        while np.any(np.abs(diff) > 0.1):
            
            for k in range(rows):  # row
                for l in range(columns):  # column
                    for a in range(5):  # iterate over actions
                        new_state = trans_model(np.array([k, l]), a)
                        Q[k, l, a] = R[k, l] + discount*V[new_state[0], new_state[1]]
            V = np.max(Q, axis=2)
            diff = V - V_old
            V_old = np.copy(V)
            
    return V, Q


##
def maxAction(V, R, discount, probModel=None):
    pass

##
def findPolicy(Q, probModel=None):
    return np.argmax(Q, axis=2)

############################
def trans_model(state, action, probModel=False):  
    if not probModel:
        if action == 0:  # down
            if state[0] < 8: 
                new_state = state + np.array([1, 0])
            else: 
                new_state = state
        
        elif action == 1:  # right
            if state[1] < 9:
                new_state = state + np.array([0, 1])
            else:
                new_state = state
        
        elif action == 3:  # left
            if state[1] > 0:
                new_state = state + np.array([0, -1])
            else:
                new_state = state
        
        elif action == 2:  # up
            if state[0] > 0:
                new_state = state + np.array([-1, 0]) 
            else:
                new_state = state
                    
        elif action == 4:  # stay
            new_state = state
        
        else:
            print('no action was given bip bup')
            new_state = state
    
    return new_state


saveFigures = True

data = genGridWorld()
grid_world = data[0]
grid_list = data[1]

# YOUR CODE HERE
probModel = ...

ax = showWorld(grid_world, 'Environment')
showTextState(grid_world, grid_list, ax)
if saveFigures:
    plt.savefig('gridworld.pdf')

# Finite Horizon
V, Q = ValIter(grid_world, 0, 15, False)
V = V[:,:,0];
showWorld(np.maximum(V, 0), 'Value Function - Finite Horizon')
if saveFigures:
    plt.savefig('value_Fin_15.pdf')

policy = findPolicy(Q)
ax = showWorld(grid_world, 'Policy - Finite Horizon')
showPolicy(policy, ax)
if saveFigures:
    plt.savefig('policy_Fin_15.pdf')

# Infinite Horizon
V, Q= ValIter(grid_world, 0.8, 15, True)
showWorld(np.maximum(V, 0), 'Value Function - Infinite Horizon')
if saveFigures:
    plt.savefig('value_Inf_08.pdf')

policy = findPolicy(Q);
ax = showWorld(grid_world, 'Policy - Infinite Horizon')
showPolicy(policy, ax)
if saveFigures:
    plt.savefig('policy_Inf_08.pdf')

# Finite Horizon with Probabilistic Transition
V, Q = ValIter(grid_world, 0, 15, infHor=False, probModel=True)
V = V[:,:,0];
showWorld(np.maximum(V, 0), 'Value Function - Finite Horizon with Probabilistic Transition')
if saveFigures:
    plt.savefig('value_Fin_15_prob.pdf')

policy = findPolicy(Q)
ax = showWorld(grid_world, 'Policy - Finite Horizon with Probabilistic Transition')
showPolicy(policy, ax)
if saveFigures:
    plt.savefig('policy_Fin_15_prob.pdf')
