import numpy as np


def multivariate_gaussian(x, mu, sigma):
    assert len(x) == len(mu) == len(sigma)

    x = np.asarray(x)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)

    n_dim = len(x)

    return 1/(2*np.pi)**(n_dim/2) * 1/(np.abs(np.linalg.det(sigma))**(1/2)) * np.exp(-0.5 * (x-mu)@np.linalg.inv(sigma)@(x-mu))


def multivariate_gaussian_log_gradient(x, mu, sigma):
    assert len(x) == len(mu) == len(sigma)

    x = np.asarray(x)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)

    partial_mu = - np.linalg.inv(sigma) @ (x-mu)
    partial_sigma = - 0.5 * np.linalg.inv(sigma) + 0.5 * np.linalg.inv(sigma)@np.outer((x-mu),(x-mu))@np.linalg.inv(sigma)

    return partial_mu, partial_sigma