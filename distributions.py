import numpy as np
from scipy.stats import norm


def lomax_tail(t, alpha, lam):
    """Survival function of Lomax(alpha, lambda)."""
    return (1 + t / lam) ** (-alpha)


def gaussian_tail(t, n, sigma):
    """P(S_n > t) for iid N(0, sigma^2) summands."""
    return 1 - norm.cdf(t, loc = 0, scale = np.sqrt(n) * sigma)
