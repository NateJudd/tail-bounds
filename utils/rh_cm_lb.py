import numpy as np
from scipy.integrate import quad

from utils.distributions import lomax_tail, lomax_cdf


def compute_I(alpha, q, lam):
    """Integral I(alpha, q, lambda) appearing in the RH-CM lower bound."""
    beta = abs(q)
    cval = lam * ((beta + 1) * (alpha + 1)) ** (-1 / alpha)

    def integrand(u):
        return (1 + cval * u ** (1 / alpha)) ** (beta * (alpha + 1)) * np.exp(-u)

    result, _ = quad(integrand, 0, np.inf)
    return result


def rh_term(t, m, n, alpha, lam, q, prefactor):
    """Riesz-Holder contribution from the truncated (n-1)-sum."""
    p = q / (q - 1)
    tail_prob = lomax_tail(t - m, alpha, lam)
    return prefactor * tail_prob ** (1 / p)


def build_target(cfg):
    alpha = cfg["alpha"]
    q     = cfg["q"]
    lam   = cfg["lambda"]
    n     = cfg["n"]

    p = q / (q - 1)
    I = compute_I(alpha, q, lam)
    prefactor = lam**alpha * I**((n - 1) / q)

    def bound(t, theta=0.5):
        m = theta * t

        OBJ      = 1 - (1 - lomax_tail(t, alpha, lam)) ** n
        P_M_le_m = lomax_cdf(m, alpha, lam) ** n
        RH       = rh_term(t, m, n, alpha, lam, q, prefactor)

        return OBJ + P_M_le_m * RH

    return bound
