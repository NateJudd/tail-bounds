import numpy as np
import matplotlib.pyplot as plt

from config import get_config
from utils.distributions import lomax_tail, gaussian_tail
from utils.simulation    import simulate_lomax, simulate_gaussian, compute_mc_prob
from utils.bounds        import compute_bounds
from utils.rh_cm_lb      import build_target
from utils.scaling       import compute_t_vals
from utils.plotting      import plot_all, plot_ratio, plot_log_ratio, plot_log_prob


def run_experiment(cfg=None):
    if cfg is None:
        cfg = get_config()

    n      = cfg["n"]
    c_vals = cfg["c_vals"]
    scaling = cfg.get("scaling", "linear")

    t_vals = compute_t_vals(c_vals, n, scaling)

    # --- simulate ---
    if cfg["dist"] == "lomax":
        S_n = simulate_lomax(n, cfg["M"], cfg["alpha"], cfg["lambda"])
    else:
        S_n = simulate_gaussian(n, cfg["M"], cfg["sigma"])

    truth = compute_mc_prob(S_n, t_vals)

    # --- target bound ---
    F = build_target(cfg)
    target_vals = np.array([F(t) for t in t_vals])

    # --- competitor bounds ---
    bounds = compute_bounds(t_vals, cfg)

    # --- plots ---
    figs = [
        plot_all(c_vals, t_vals, truth, target_vals, bounds, cfg),
        plot_ratio(c_vals, truth, target_vals, bounds, cfg),
        plot_log_ratio(c_vals, truth, target_vals, bounds, cfg),
        plot_log_prob(c_vals, truth, target_vals, bounds, cfg),
    ]
    plt.show()
    return figs

