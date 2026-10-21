"""Confidence bounds for the mean score of a candidate on a *finite* test set.

We sample test examples without replacement, so the quantity of interest is the
exact mean over the whole test set (what a full evaluation would report). Two
classical finite-population bounds apply (Bardenet & Maillard, 2015):

* Hoeffding–Serfling: radius sqrt(rho_n * L / (2 n))
* Empirical Bernstein–Serfling: sigma_hat * sqrt(2 rho_n L / n) + kappa * L / n

where rho_n shrinks to 0 as n approaches N, so once a candidate has been run on
every example its interval collapses to the exact score. We use the smaller of
the two radii, each at confidence delta / 2.

L includes a union bound over all candidates and all sample sizes, so the
intervals hold *simultaneously and at every step* with probability >= 1 - delta.
That is what makes adaptive stopping safe.
"""
from __future__ import annotations

import math

import numpy as np

KAPPA = 7.0 / 3.0 + 3.0 / math.sqrt(2.0)


def serfling_rho(n: np.ndarray, N: int) -> np.ndarray:
    n = np.asarray(n, dtype=float)
    safe = np.maximum(n, 1.0)
    rho = np.minimum(1.0 - (safe - 1.0) / N, (1.0 - safe / N) * (1.0 + 1.0 / safe))
    return np.clip(rho, 0.0, 1.0)


def union_log(n: np.ndarray, n_candidates: int, delta: float) -> np.ndarray:
    """log(1 / delta_n) for a two-sided bound, union over candidates and sample sizes."""
    n = np.maximum(np.asarray(n, dtype=float), 1.0)
    return np.log(math.pi ** 2 * n_candidates * n * n / (3.0 * delta))


def radius(n: np.ndarray, sum_x: np.ndarray, sum_x2: np.ndarray, N: int,
           n_candidates: int, delta: float) -> np.ndarray:
    """Half-width of the confidence interval for each candidate (scores in [0, 1])."""
    n = np.asarray(n, dtype=float)
    out = np.full(n.shape, np.inf)
    seen = n > 0
    if not seen.any():
        return out
    ns = n[seen]
    rho = serfling_rho(ns, N)
    log_term = union_log(ns, n_candidates, delta / 2.0)
    hoeffding = np.sqrt(rho * log_term / (2.0 * ns))
    mean = sum_x[seen] / ns
    var = np.maximum(sum_x2[seen] / ns - mean ** 2, 0.0)
    # Their theorem holds with probability 1 - 5d and uses log(5/d); taking d = delta_n / 5
    # gives failure probability delta_n and log term log(25 / delta_n) (deliberately conservative).
    bernstein_log = log_term + math.log(25.0)
    bernstein = np.sqrt(var) * np.sqrt(2.0 * rho * bernstein_log / ns) + KAPPA * bernstein_log / ns
    r = np.minimum(hoeffding, bernstein)
    r[ns >= N] = 0.0  # evaluated on the whole test set: the score is exact
    out[seen] = r
    return out
