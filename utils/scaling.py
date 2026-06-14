import numpy as np


def compute_t_vals(c_vals, n, scaling):
    """Map c-grid to t-grid under the chosen scaling regime."""
    c = np.asarray(c_vals)
    if scaling == "constant":
        return c
    elif scaling == "sqrt":
        return c * np.sqrt(n)
    elif scaling == "linear":
        return c * n
    else:
        raise ValueError(f"Unknown scaling '{scaling}'")
