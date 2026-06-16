"""
demo.py — end-to-end examples for the tail_bounds project.

Run from the repo root:
    python examples/demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from config import get_config
from main   import run_experiment


# ─── Example 1: default config ────────────────────────────────────────────────
# Lomax(alpha=2), n=5, linear scaling. Good sanity check — bound should
# track the truth reasonably well across the c-grid.

print("Example 1: default config")
cfg = get_config()
run_experiment(cfg)


# ─── Example 2: heavier tail ──────────────────────────────────────────────────
# alpha < 1 puts us in the infinite-mean regime. The one-big-jump heuristic
# dominates and the RH-CM bound should tighten relative to Bonferroni.

print("Example 2: heavy tail (alpha=0.8, n=3)")
cfg = get_config()
cfg["alpha"]   = 0.8
cfg["n"]       = 3
cfg["q"]       = -70.0
cfg["c_vals"]  = np.linspace(0.05, 5, 100)
run_experiment(cfg)


# ─── Example 3: scaling regimes ───────────────────────────────────────────────
# Same distribution, three different t_n scalings side by side.

print("Example 3: scaling regimes")
for scaling in ("constant", "sqrt", "linear"):
    cfg = get_config()
    cfg["scaling"] = scaling
    cfg["c_vals"]  = np.linspace(0.05, 3, 80)
    print(f"  scaling = {scaling}")
    run_experiment(cfg)


# ─── Example 4: larger n ──────────────────────────────────────────────────────
# Check how the bound holds up as n grows — the (n-1)/q exponent in the
# prefactor means it scales with n, so worth verifying numerically.

print("Example 4: n = 10")
cfg = get_config()
cfg["n"]      = 10
cfg["alpha"]  = 1.5
cfg["q"]      = -2.0
cfg["c_vals"] = np.linspace(0.05, 4, 100)
run_experiment(cfg)


plt.show()
