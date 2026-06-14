# Tail Bounds for Heavy-Tailed Sums

A comparison of lower bounding methods.

Numerical comparison of tail probability bounds for sums of i.i.d. random variables, with a focus on the Lomax (Pareto Type II) distribution. Implements a Riesz–Hölder / Chebyshev–Markov lower bound alongside classical competitors (Bonferroni, Chernoff, Bernstein, Bennett, Paley–Zygmund).

## Structure

```
.
├── main.py               # entry point / experiment runner
├── config.py             # global parameters
├── requirements.txt
└── utils/
    ├── distributions.py  # tail / CDF functions
    ├── simulation.py     # Monte Carlo samplers
    ├── bounds.py         # upper and lower bounds
    ├── rh_cm_lb.py       # RH-CM target bound
    ├── scaling.py        # t_n = c, c*sqrt(n), c*n regimes
    └── plotting.py       # matplotlib figures
```

## Quickstart

```bash
pip install -r requirements.txt
python main.py
```

The default run uses a Lomax(alpha=0.9) distribution with n=3 summands and a linear scaling regime. Four diagnostic plots are produced: raw tail probabilities, approximation/truth ratio, log-ratio, and log-probability.

## Configuration

Edit `config.py` or pass a modified dict directly to `run_experiment`:

```python
from config import get_config
from main import run_experiment

cfg = get_config()
cfg["n"]     = 10
cfg["alpha"] = 1.5
cfg["dist"]  = "lomax"
run_experiment(cfg)
```

Key parameters:

| Parameter | Description |
|-----------|-------------|
| `n`       | number of summands |
| `alpha`   | Lomax tail index |
| `lambda`  | Lomax scale |
| `q`       | Hölder exponent (negative → conjugate pair) |
| `M`       | MC sample size |
| `scaling` | `"constant"`, `"sqrt"`, or `"linear"` |
| `dist`    | `"lomax"` or `"gaussian"` |
