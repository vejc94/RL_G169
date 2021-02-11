from matplotlib import pyplot as plt
import numpy as np
from Pend2dBallThrowDMP import *
import time
# %matplotlib inline
np.set_printoptions(precision=3, linewidth=100000)


env = Pend2dBallThrowDMP()

numDim = 10
numSamples = 25
maxIter = 10
numTrials = 10

# YOUR CODE HERE
from gaussian_functions import *
mu = np.zeros(numDim)
sigma = np.diag((np.ones(10)))
alpha = 0.1

# reward = np.zeros((maxIter, numSamples))
# for i in range(maxIter):
#     gradient_mu = np.zeros(mu.shape)
#     for j in range(numSamples):
#         theta = np.random.multivariate_normal(mu, sigma)
#         reward[i,j] = env.getReward(theta)
#         gradient_mu += multivariate_gaussian_log_gradient(theta, mu, sigma)[0] * reward[i,j]
#
#     mu = mu + alpha * gradient_mu


reward = np.zeros((maxIter, numSamples))
for i in range(maxIter):
    gradient_mu = np.zeros(mu.shape)
    theta = [None] * numSamples
    for j in range(numSamples):
        theta[j] = np.random.multivariate_normal(mu, sigma)
        reward[i,j] = env.getReward(theta[j])

    b = np.mean(reward[i, :])
    for j in range(numSamples):
        gradient_mu += multivariate_gaussian_log_gradient(theta[j], mu, sigma)[0] * (reward[i,j] - b)

    mu = mu + alpha * gradient_mu

fig0, ax0 = plt.subplots()
ax0.plot(np.arange(maxIter), [np.mean(reward[i]) for i in range(maxIter)])
plt.show()
