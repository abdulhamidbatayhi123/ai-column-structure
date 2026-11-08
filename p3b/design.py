"""Design sets: scrambled Sobol points in the pre-registered boxes (PREREG_P3B_draft.md, Section 4).

Each set is drawn from its own seed (the seeds are literals in scripts/design_sets.py). A set of
`count` points takes the first `count` of 2^m scrambled Sobol points, 2^m >= count, mapped
linearly onto the box. Train subsets are prefixes of the train pool, so 16 ⊂ 32 ⊂ ... ⊂ 256 and
every power-of-two subset is itself a balanced Sobol design.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import qmc

_A = {"RH_ads": (0.20, 0.60), "T_ads": (293.0, 308.0), "T_des": (333.0, 363.0),
      "v": (0.03, 0.30), "alpha": (0.5, 1.2), "beta": (0.5, 1.0)}
BOXES: dict[str, dict[str, tuple[float, float]]] = {
    "A": _A,
    "A_OOD": dict(_A, RH_ads=(0.10, 0.20), T_des=(363.0, 383.0)),
    "B": {"T_hot": (343.0, 363.0), "T_cold": (293.0, 303.0), "v": (0.5e-3, 3.0e-3),
          "alpha": (0.4, 0.9), "beta": (0.6, 1.0)},
}

_COMMON = {"train": (256, 6), "validation": (32, 6), "long_validation": (4, 60), "long_test": (48, 60)}
SET_SIZES: dict[tuple[str, str], tuple[int, int | None]] = {
    ("A_primary", "pilot_train"): (32, 6),
    ("A_primary", "pilot_val"): (16, 60),
    **{("A_primary", k): v for k, v in _COMMON.items()},
    ("A_primary", "ood"): (24, 30),
    # H6 operating points (RH_ads, T_ads, v); cycles set later by the cyclic-steady-state rule
    ("A_primary", "scenarios"): (8, None),
    **{(s, k): v for s in ("A_replication", "B") for k, v in {"pilot": (8, 60), **_COMMON}.items()},
}

_SCENARIO_KEYS = ("RH_ads", "T_ads", "v")


def _box_and_keys(system: str, set_name: str):
    if system == "B":
        box = BOXES["B"]
    elif set_name == "ood":
        box = BOXES["A_OOD"]
    else:
        box = BOXES["A"]
    keys = _SCENARIO_KEYS if set_name == "scenarios" else tuple(box)
    return box, keys


def make_set(system: str, set_name: str, seed: int) -> list[dict]:
    """The points of one (system, set): sampled quantities plus "index" and "n_cycles"."""
    count, n_cycles = SET_SIZES[(system, set_name)]  # KeyError for an unknown pair
    box, keys = _box_and_keys(system, set_name)
    m = max(0, math.ceil(math.log2(count)))
    u = qmc.Sobol(d=len(keys), scramble=True, seed=seed).random_base2(m)[:count]
    lo = np.array([box[k][0] for k in keys])
    hi = np.array([box[k][1] for k in keys])
    x = lo + u * (hi - lo)
    return [{**{k: float(v) for k, v in zip(keys, row)}, "index": i, "n_cycles": n_cycles}
            for i, row in enumerate(x)]


def chunk_indices(n_traj: int, n_chunks: int) -> list[list[int]]:
    """Contiguous chunks of range(n_traj) whose sizes differ by at most one."""
    if not 0 < n_chunks <= n_traj:
        raise ValueError(f"need 0 < n_chunks <= n_traj, got {n_chunks}, {n_traj}")
    base, extra = divmod(n_traj, n_chunks)
    out, start = [], 0
    for k in range(n_chunks):
        size = base + (k < extra)
        out.append(list(range(start, start + size)))
        start += size
    return out


def train_subset(pool: list[dict], k: int) -> list[dict]:
    """The first k points of the train pool (nested subsets)."""
    if not 0 < k <= len(pool):
        raise ValueError(f"subset size {k} outside 1..{len(pool)}")
    return pool[:k]
