"""Power of the full decision rules (PREREG Section 9; Plan 4, Task 3).

Synthetic crossed data: the log error ratio of trajectory t and seed s is
    v[t, s] = log(ratio) + a_t + b_s + e_ts,
with a_t ~ N(0, sd_traj^2) and the seed component split into a part shared across trajectories
(b_s, the same trained models on every trajectory) and a trajectory-specific part (e_ts):
var(b) = seed_shared * sd_seed^2, var(e) = (1 - seed_shared) * sd_seed^2. Each replicate runs the
pre-registered rule exactly: the seed-averaged values, the one-sided signed-rank test at alpha_h,
the (1 - 2 alpha_h) pigeonhole bootstrap interval of HL, and the directional words. The fraction of
each word is returned. The inputs are the pilot's upper-80 % SDs (pilot rules).
"""
from __future__ import annotations

import math

import numpy as np

from p3b.stats.tests import hodges_lehmann, pigeonhole_bootstrap, signed_rank
from p3b.stats.words import directional_word

WORDS = ("SUPPORTED", "SMALL", "REFUTED", "REFUTED (REVERSED)", "INCONCLUSIVE")


def crossed_values(log_ratio, sd_traj, sd_seed, n_traj, n_seed, rng, seed_shared: float = 0.5):
    a = rng.normal(0.0, sd_traj, size=(n_traj, 1))
    b = rng.normal(0.0, sd_seed * math.sqrt(seed_shared), size=(1, n_seed))
    e = rng.normal(0.0, sd_seed * math.sqrt(1.0 - seed_shared), size=(n_traj, n_seed))
    return log_ratio + a + b + e


def simulate(log_ratio, sd_traj, sd_seed, n_traj: int = 48, n_seed: int = 5, reps: int = 1000,
             draws: int = 10_000, seed: int = 0, alpha_h: float = 0.05 / 3, seed_shared: float = 0.5) -> dict:
    rng = np.random.default_rng(seed)
    counts = dict.fromkeys(WORDS, 0)
    for r in range(reps):
        v = crossed_values(log_ratio, sd_traj, sd_seed, n_traj, n_seed, rng, seed_shared)
        x = v.mean(axis=1)
        p, _ = signed_rank(x, alternative="less")
        boot = pigeonhole_bootstrap(v, hodges_lehmann, draws=draws, seed=int(rng.integers(2**31)))
        lo, hi = np.percentile(boot, [100 * alpha_h, 100 * (1 - alpha_h)])
        a_holds = p <= alpha_h
        b_holds = hi < 0.0
        word = directional_word(a_holds, b_holds, math.exp(hodges_lehmann(x)), (math.exp(lo), math.exp(hi)))
        counts[word] += 1
    return {w: c / reps for w, c in counts.items()}
