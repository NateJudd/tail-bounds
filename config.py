import numpy as np

def get_config():
    return {
        "n": 5,
        "alpha": 2.0,
        "lambda": 1.0,
        "sigma": 1.0,
        "q": -0.5,
        "M": 100_000,
        "c_vals": np.linspace(0.05, 2, 60),
        "dist": "lomax",       # or "gaussian"
        "scaling": "linear",   # "constant", "sqrt", "linear"
    }
