import numpy as np


def simulate_lomax(n, M, alpha, lam):
    """Draw M realisations of S_n = X_1 + ... + X_n, X_i ~ Lomax(alpha, lam).

    Uses the inverse-CDF: X = lam * (U^{-1/alpha} - 1), U ~ Uniform(0,1).
    """
    U = np.random.uniform(size=(M, n))
    X = lam * ((1 - U) ** (-1 / alpha) - 1)
    return X.sum(axis=1)


def simulate_gaussian(n, M, sigma):
    """Draw M realisations of S_n for iid N(0, sigma^2) summands."""
    X = np.random.normal(scale=sigma, size=(M, n))
    return X.sum(axis=1)


def compute_mc_prob(S_n, t_vals):
    """Monte Carlo estimate of P(S_n > t) for each t in t_vals."""
    return np.array([(S_n > t).mean() for t in t_vals])
