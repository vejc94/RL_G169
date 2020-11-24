# This is one of the two main classes you need to run.
#
# CTLS is a list of cells with your controller name
# ('JacTrans','JacPseudo,'JacDPseudo','JacNullSpace'). You can run more controllers
# one after another, e.g. passing {'JacTrans','JacPseudo'}.
#
# PAUSETIME is the number of seconds between each iteration for the
# animated plot. If 0 only the final position of the robot will be
# displayed.

import numpy as np
from simSys import *
from DoubleLink import *


def taskCtlComp(ctls=['JacDPseudo'], pauseTime=False, resting_pos=None):
    dt = 0.002
    robot = DoubleLink()
    robot.friction = np.array([2.5, 2.5])
    t_end = 3.0
    time = np.arange(0, t_end, dt)
    nSteps = len(time)
    numContrlComp = len(ctls)
    target = {}
    target['x'] = np.tile([-0.35, 1.5], (nSteps, 1))
    target['cartCtl'] = True
    states = simSys(robot, dt, nSteps, ctls, target, pauseTime, resting_pos)

    colors = [1, 2, 3]
    taskSpace_plot(states, numContrlComp, ctls, time, robot)
    traj_plot(states, colors, numContrlComp, ctls, time, plotVel=0)
    # traj_plot(states, colors, numContrlComp, ctls, time, plotVel=1)


def taskSpace_plot(states, numContrlComp, ctls, time, robot: DoubleLink):
    from matplotlib.lines import Line2D

    fig = plt.figure()
    ax = fig.add_subplot(111)

    x1 = np.empty((states.shape[0], 2))
    x2 = np.empty((states.shape[0], 2))
    for i in range(states.shape[0]):
        x1[i,:], x2[i,:] = robot.getJointsInTaskSpace(q=states[i,:])

    # lineX1 = Line2D(x1[:,0], x1[:,1])
    # lineX2 = Line2D(x2[:,0], x2[:,1])
    # ax.add_line(lineX1, label="X1")
    # ax.add_line(lineX2, label="X2")
    ax.plot(x1[:,0], x1[:,1], label="X1")
    ax.plot(x2[:,0], x2[:,1], label="X2")

    #plt.legend(tuple(names))
    plt.xlabel('x', fontsize=15)
    plt.ylabel('y', fontsize=15)

    plt.grid()
    plt.show()
    # plt.savefig(fname="daniel/SavedPlots/" + "TaskCtl_" + "Velocity"*plotVel + "Position"*(1-plotVel) + "_Joint_"
    #                   + str(statei) + ".pdf", format='pdf')


def traj_plot(states, colors, numContrlComp, ctls, time, plotVel):

    stateNo = (1, 2)

    for statei in stateNo:
        plt.figure()

        names = []

        for k in range(numContrlComp):
            names += [ctls[k] + '_' + str(statei)]
            plt.plot(time, states[k*len(time):(k+1)*len(time), plotVel+2*(statei-1)::4], linewidth=2)

        plt.legend(tuple(names))
        plt.xlabel('time in s', fontsize=15)
        plt.ylabel('angle in rad', fontsize=15)

        if plotVel:
            plt.title('Velocity for Joint ' + str(statei), fontsize=20)
        else:
            plt.title('Position for Joint ' + str(statei), fontsize=20)

        plt.xlim(0,3)
        plt.grid()
        plt.savefig(fname="daniel/SavedPlots/" + "TaskCtl_" + "Velocity"*plotVel + "Position"*(1-plotVel) + "_Joint_"
                          + str(statei) + ".pdf", format='pdf')