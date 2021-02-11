import numpy as np
import matplotlib.pyplot as plt
from gaussian_functions import *

x = np.zeros((1000,2))
x[:,0] = np.linspace(-20, 20, 1000)
mu = np.array((0,0))
sigma = np.array([[20,0],[0,1]])
y = np.zeros(1000)
mu_der = np.zeros((1000,2))
for i in range(1000):
    y[i] = multivariate_gaussian(x[i], mu, sigma)
    mu_der[i] = multivariate_gaussian_log_gradient(x[i], mu, sigma)[0]

fig0, ax0 = plt.subplots()
ax0.plot(x[:,0],y)
plt.show()

fig0, ax0 = plt.subplots()
ax0.plot(x[:,0],mu_der[:,0])
plt.show()
