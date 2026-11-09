"""The pilot's rules (PREREG Section 10; Plan 4, Task 1). Each takes pilot quantities and returns
only the chosen value, so the blinded script (scripts/pilot_rules.py) never needs to print an error
value by arm. A rule with no admissible value returns None (or "demoted"), never a silent default.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as st

LR_GRID = (3e-4, 1e-3, 3e-3)
IMPROVEMENT = 0.02            # per 1,000 iterations
H_RATE = 1e-4                 # S's entropy-violation rate (fraction of steps)
DEFECT_CAP = 1e-3
K_TOL = 0.01
POWER = 0.8


def choose_lr(losses: dict) -> dict:
    """{arm: {lr: validation rollout loss over one cycle after 2,000 iterations}} -> {arm: lr};
    the smallest loss, ties to the smaller learning rate; None if no run finished."""
    out = {}
    for arm, by_lr in losses.items():
        ok = {lr: v for lr, v in by_lr.items() if v == v and math.isfinite(v)}
        out[arm] = min(sorted(ok), key=lambda lr: ok[lr]) if ok else None
    return out


def choose_iterations(curves: dict):
    """{arm: [(iteration, pooled validation loss), ...] every 1,000} -> the smallest checkpoint at
    which every arm improved by less than 2 % over the previous 1,000 iterations; None if none."""
    checkpoints = sorted({i for c in curves.values() for i, _ in c})
    for i in checkpoints:
        ok = True
        for c in curves.values():
            d = dict(c)
            if i not in d or (i - 1000) not in d:
                ok = False
                break
            prev, now = d[i - 1000], d[i]
            if not (prev > 0 and (prev - now) / prev < IMPROVEMENT):
                ok = False
                break
        if ok:
            return i
    return None


def choose_h(viol_rates: dict, stiff_ok: dict):
    """The largest h in {T/200, T/400, T/800} with S's entropy-violation rate below 1e-4 of steps and
    the near-dry stiffness check passed; None if none passes."""
    ok = [h for h in viol_rates if viol_rates[h] < H_RATE and stiff_ok.get(h, False)]
    return max(ok) if ok else None


def thresholds(floors: dict, neg_production, g2, defects_per_step, tau_ref: float) -> dict:
    """Per system and material: violation thresholds (10 x the reference floors), tau_S (10 x the
    99.9th percentile of the magnitude of S's negative production on the pilot), kappa (1e-3 x the
    median |P grad S|^2), the projection-defect bound (10 x the 99.9th percentile of defect steps
    per step, capped at 1e-3 of steps) and tau_ref (from the reference gate, passed through)."""
    out = {f"violation_{k}": 10.0 * v for k, v in floors.items()}
    neg = np.abs(np.asarray(neg_production, dtype=float))
    out["tau_S"] = 10.0 * float(np.percentile(neg, 99.9)) if neg.size else 0.0
    out["kappa"] = 1e-3 * float(np.median(np.asarray(g2, dtype=float)))
    out["defect_bound"] = min(10.0 * float(np.percentile(np.asarray(defects_per_step, float), 99.9)), DEFECT_CAP)
    out["tau_ref"] = float(tau_ref)
    return out


def choose_K(eps_by_K: dict):
    """The smallest K in {1, 2, 4, 8} at which C+L with unit multipliers reproduces C's epsilon_W
    within 1 %; None if none does."""
    ok = [K for K in sorted(eps_by_K) if eps_by_K[K] <= K_TOL]
    return ok[0] if ok else None


def _upper80(s: float, df: int) -> float:
    return s * math.sqrt(df / st.chi2.ppf(0.2, df))


def upper80_sd(values) -> tuple:
    """values (n_traj, n_seed) of the log error ratio -> the upper 80 % confidence bounds of the
    trajectory SD (of the seed-averaged values, df n_traj - 1) and of the seed SD (pooled within
    trajectories, df n_traj (n_seed - 1))."""
    v = np.asarray(values, dtype=float)
    n_t, n_s = v.shape
    s_traj = float(v.mean(axis=1).std(ddof=1))
    s_seed = float(np.sqrt(v.var(axis=1, ddof=1).mean())) if n_s > 1 else 0.0
    return _upper80(s_traj, n_t - 1), (_upper80(s_seed, n_t * (n_s - 1)) if n_s > 1 else 0.0)


def h4_margin(power_by_m: dict):
    """The smallest margin in {1.10, ..., 1.20} reaching power 0.8 at the Holm worst case; else
    "demoted" (H4 becomes secondary)."""
    ok = [m for m in sorted(power_by_m) if power_by_m[m] >= POWER and m <= 1.20 + 1e-12]
    return ok[0] if ok else "demoted"
