import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


_SCALING_LABELS = {
    "constant": r"$t_n = c$",
    "sqrt":     r"$t_n = c\sqrt{n}$",
    "linear":   r"$t_n = cn$",
}

EPS = 1e-12


def _scaling_label(scaling):
    return _SCALING_LABELS.get(scaling, "unknown")


def _build_bounds_df(bounds, c_vals, transform=None):
    """Collect bound arrays into a list of (name, values) pairs, applying an
    optional transform and stripping non-finite entries."""
    out = []
    for name, vals in bounds.items():
        vals = np.asarray(vals, dtype=float)
        if len(vals) != len(c_vals):
            continue
        if transform is not None:
            vals = transform(vals)
        vals[~np.isfinite(vals)] = np.nan
        out.append((name, vals))
    return out


def _base_fig(cfg, title, ylabel):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f"{title}  (n = {cfg['n']})")
    ax.set_xlabel("c")
    ax.set_ylabel(ylabel)
    subtitle = _scaling_label(cfg["scaling"])
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
            ha="center", fontsize=9, color="grey")
    return fig, ax


def plot_all(c_vals, t_vals, truth, target, bounds, cfg):
    """Raw tail probabilities — truth, target bound, and competitors."""
    fig, ax = _base_fig(cfg, "Tail probability comparison", "Probability")

    ax.plot(c_vals, truth,  label="Truth",  lw=2)
    ax.plot(c_vals, target, label="Target", lw=2, ls="--")

    for name, vals in _build_bounds_df(bounds, c_vals):
        ax.plot(c_vals, vals, label=name, lw=1.5, ls=":")

    ax.legend()
    fig.tight_layout()
    return fig


def plot_ratio(c_vals, truth, target, bounds, cfg):
    """Approximation / truth ratio; dashed line at 1."""
    fig, ax = _base_fig(cfg, "Approximation / truth", "Ratio")

    truth_safe = np.maximum(truth, EPS)

    ax.plot(c_vals, target / truth_safe, label="Target", lw=2, ls="--")
    for name, vals in _build_bounds_df(bounds, c_vals, transform=lambda v: v / truth_safe):
        ax.plot(c_vals, vals, label=name, lw=1.5, ls=":")

    ax.axhline(1, color="black", ls="dashed", lw=0.8)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_log_ratio(c_vals, truth, target, bounds, cfg):
    """Log(approximation / truth); reference line at 0."""
    fig, ax = _base_fig(cfg, "Log ratio", "log ratio")

    truth_safe = np.maximum(truth, EPS)

    def _log_ratio(v):
        return np.log(np.maximum(v / truth_safe, EPS))

    ax.plot(c_vals, _log_ratio(target), label="Target", lw=2, ls="--")
    for name, vals in _build_bounds_df(bounds, c_vals, transform=_log_ratio):
        ax.plot(c_vals, vals, label=name, lw=1.5, ls=":")

    ax.axhline(0, color="black", ls="dashed", lw=0.8)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_log_prob(c_vals, truth, target, bounds, cfg):
    """Log-scale tail probabilities."""
    fig, ax = _base_fig(cfg, "Log probability", "log probability")

    ax.plot(c_vals, np.log(np.maximum(truth,  EPS)), label="Truth",  lw=2)
    ax.plot(c_vals, np.log(np.maximum(target, EPS)), label="Target", lw=2, ls="--")

    def _log(v):
        return np.log(np.maximum(v, EPS))

    for name, vals in _build_bounds_df(bounds, c_vals, transform=_log):
        ax.plot(c_vals, vals, label=name, lw=1.5, ls=":")

    ax.legend()
    fig.tight_layout()
    return fig
