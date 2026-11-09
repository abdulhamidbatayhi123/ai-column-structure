"""The pre-registered statistics (PREREG Section 6; Plan 3, Task 6).

The unit is a test trajectory; its value is the seed-averaged log error ratio arm/U.
- hodges_lehmann: the median of the Walsh averages, the parameter the signed-rank test targets.
- signed_rank: one-sided Wilcoxon signed-rank test with Pratt's handling of zeros (zeros are
  ranked with the rest and then dropped from the statistic), always from the exact conditional
  null distribution (ties included), by a subset-sum recursion over the doubled ranks. The number
  of zeros is returned and reported.
- pigeonhole_bootstrap: seeds are crossed with trajectories, so each draw resamples the
  trajectories and the seeds independently, once each, and recomputes the seed averages from
  the drawn seeds (Owen, Ann. Appl. Stat. 1, 386-411, 2007).
- tost: two paired one-sided t-tests on the log ratios with margin +-log m.
- holm: step-down Holm; alpha_h = alpha / (k - r + 1) for the r-th smallest p; for a
  hypothesis that step-down did not reach, the level it would have had if step-down continued.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as st


def hodges_lehmann(x) -> float:
    x = np.asarray(x, dtype=float)
    i, j = np.triu_indices(x.size)
    return float(np.median((x[i] + x[j]) / 2.0))


def _signed_rank_null(weights):
    """Exact null distribution of T+ = sum of the weights with a positive sign, every sign pattern
    equally likely: a subset-sum recursion over integer weights (counts exact in float64 up to
    2^53 patterns, so n <= 52)."""
    total = int(sum(weights))
    dist = np.zeros(total + 1)
    dist[0] = 1.0
    for w in weights:
        shifted = np.zeros_like(dist)
        shifted[w:] = dist[:total + 1 - w]
        dist = dist + shifted
    return dist / dist.sum()


def signed_rank(x, alternative: str = "less"):
    """(p, n_zeros) of the one-sided signed-rank test with Pratt's zero handling, from the exact
    conditional null distribution (zeros ranked with the rest, then dropped; tied ranks allowed).
    "less" tests a median below 0 (S better for log error ratios S/U)."""
    x = np.asarray(x, dtype=float)
    n0 = int(np.sum(x == 0))
    nz = x != 0
    if not nz.any():
        return 1.0, n0
    if nz.sum() > 52:
        raise ValueError("exact signed-rank null limited to 52 non-zero values")
    ranks = st.rankdata(np.abs(x))[nz]
    w = np.rint(2 * ranks).astype(int)                       # tied ranks are multiples of 1/2
    t_obs = int(w[x[nz] > 0].sum())
    dist = _signed_rank_null(w)
    if alternative == "less":
        p = dist[:t_obs + 1].sum()
    elif alternative == "greater":
        p = dist[t_obs:].sum()
    else:
        raise ValueError(f"alternative {alternative!r}")
    return float(min(1.0, p)), n0


def pigeonhole_bootstrap(values, stat, draws: int = 10_000, seed: int = 0) -> np.ndarray:
    """values (n_traj, n_seed): per-trajectory, per-seed log ratios. Returns `draws` values of
    stat(seed-averaged values) under independent resampling of trajectories and seeds."""
    v = np.asarray(values, dtype=float)
    n_t, n_s = v.shape
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for b in range(draws):
        ti = rng.integers(0, n_t, n_t)
        si = rng.integers(0, n_s, n_s)
        out[b] = stat(v[np.ix_(ti, si)].mean(axis=1))
    return out


def tost(x, margin: float, alpha: float) -> dict:
    """Equivalence of the mean log ratio to 0 within +-margin by two one-sided paired t-tests."""
    x = np.asarray(x, dtype=float)
    n = x.size
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(n)
    p_lower = 1.0 - st.t.cdf((m + margin) / se, n - 1)      # H0: mean <= -margin
    p_upper = st.t.cdf((m - margin) / se, n - 1)            # H0: mean >= +margin
    half = st.t.ppf(1 - alpha, n - 1) * se
    return {"equivalent": bool(max(p_lower, p_upper) < alpha), "p": float(max(p_lower, p_upper)),
            "interval": (float(m - half), float(m + half)), "mean": float(m)}


def holm(pvals: dict, alpha: float = 0.05) -> dict:
    """{name: (alpha_h, rejected)} by step-down Holm."""
    order = sorted(pvals, key=lambda k: pvals[k])
    k = len(order)
    out, stopped = {}, False
    for r, name in enumerate(order, start=1):
        a_h = alpha / (k - r + 1)
        rej = (not stopped) and pvals[name] <= a_h
        stopped = stopped or not rej
        out[name] = (a_h, rej)
    return out
