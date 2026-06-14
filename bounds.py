import numpy as np
from scipy.stats import norm
from math import comb


# ─── heavy-tail bounds ────────────────────────────────────────────────────────

def lb_onejump(t, n, alpha, lam):
    """Lower bound via one-big-jump heuristic."""
    return 1 - (1 - (1 + t / lam) ** (-alpha)) ** n


def lb_bonf2(t, n, alpha, lam):
    """Bonferroni-order-2 lower bound."""
    p = (1 + t / lam) ** (-alpha)
    return np.maximum(n * p - comb(n, 2) * p ** 2, 0)


# ─── light-tail bounds ────────────────────────────────────────────────────────

def gaussian_exact(t, n, sigma):
    """Exact Gaussian tail (benchmark, approximate, but not a bound)."""
    return 1 - norm.cdf(t, loc=0, scale = np.sqrt(n) * sigma)


def chernoff_gaussian(t, n, sigma):
    """Chernoff bound for sub-Gaussian sums: exp(-t^2 / (2 n sigma^2))."""
    return np.exp(- t ** 2 / (2 * n * sigma ** 2))


def bernstein_bound(t, n, sigma, b):
    """Bernstein bound; b is the scale parameter (e.g. b = sigma as rough proxy)."""
    return np.exp( - t ** 2 / (2 * n * sigma ** 2 + (2 / 3) * b * t))


def bennett_bound(t, n, sigma, b):
    """Bennett bound — sharper than Bernstein for bounded RVs"""
    x = t / (n * sigma ** 2)
    h = (1 + x) * np.log1p(x) - x
    return np.exp(- h * ( n * sigma ** 2 / ( b ** 2 ) ) )


# ─── light-tail lower bounds ──────────────────────────────────────────────────

def lb_paley_zygmund(t, n, sigma):
    """Paley–Zygmund lower bound applied to S_n^2."""
    var = n * sigma ** 2
    ES2 = var
    ES4 = 3 * var ** 2   # Gaussian fourth moment

    t = np.asarray(t, dtype=float)
    theta = t ** 2 / ES2

    val = (1 - theta) ** 2 * (ES2 ** 2 / ES4)
    val = np.where(theta >= 1, np.nan, val)
    val = np.where(val < 0, 0.0, val)
    return val


def clt_approx(t, n, sigma):
    """CLT approximation P(S_n > t) ≈ 1 - Phi(t / (sqrt(n) sigma))."""
    return 1 - norm.cdf(t / ( np.sqrt(n) * sigma ) )


# ─── unified interface ────────────────────────────────────────────────────────

def compute_bounds(t_vals, cfg):
    """Compute whichever bounds are relevant for the configured distribution."""
    n = cfg["n"]
    results = {}

    if cfg["dist"] == "lomax":
        results["onejump"] = np.array([lb_onejump(t, n, cfg["alpha"], cfg["lambda"]) for t in t_vals])
        results["bonf2"]   = np.array([lb_bonf2(t, n, cfg["alpha"], cfg["lambda"]) for t in t_vals])

    elif cfg["dist"] == "gaussian":
        sigma = cfg["sigma"]
        t = np.asarray(t_vals)
        results["exact"]     = gaussian_exact(t, n, sigma)
        results["chernoff"]  = chernoff_gaussian(t, n, sigma)
        results["pz"]        = lb_paley_zygmund(t, n, sigma)
        results["clt"]       = clt_approx(t, n, sigma)
        results["bernstein"] = bernstein_bound(t, n, sigma, b=sigma)

    return results
