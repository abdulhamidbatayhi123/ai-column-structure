"""The training budget from measured per-iteration costs (docs/compute_3B.md, Sections 2 and 3).

Training runs per arm come from the PREREG design table and the descriptive sensitivities; the cost of
one run is seconds per iteration (one thread, `results/timing_arms.json`, Kaggle) times the iteration
count, both chosen in the pilot (PREREG 10: one iteration count for every arm, h RK4 steps per
snapshot interval). h = 4 was not timed: it is extrapolated linearly through h = 1 and 2 (cost = a + b h).
U-long trains for S's wall-clock. C+L is not a latent arm and was not timed: its hours are an input,
reported as unmeasured. The reduction order is PREREG Section 11 / compute_3B Section 3.

    python scripts/budget.py [--timing results/timing_arms.json] [--iterations 8000] [--h 2]
"""
from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# seeds per design cell (PREREG design table); learning curve = 3 sizes (16, 64, 128) x 3 seeds
DESIGN = {
    "U":      {"A_full": 5, "A_32": 5, "A_lc": 9, "Arep": 6, "B": 10},
    "U+F":    {"A_full": 5, "A_32": 5, "Arep": 6, "B": 10},
    "U-long": {"A_full": 3},
    "P":      {"A_full": 5, "A_32": 5, "A_lc": 9, "Arep": 6},
    "S":      {"A_full": 5, "A_32": 5, "A_lc": 9, "Arep": 6, "B": 10},
    "S-L":    {"A_full": 5, "A_32": 5, "B": 6},
    "S-M":    {"A_full": 5},
    "S-Pi":   {"A_full": 5},
    "FNO":    {"A_full": 3, "A_32": 3},
    "C+L":    {"A_full": 3, "A_32": 3},
}
# descriptive sensitivities (PREREG 6): U, U+F, S, 3 seeds each
SENSITIVITIES = {"degraded": {"U": 3, "U+F": 3, "S": 3}, "latent_dim": {"U": 3, "U+F": 3, "S": 3}}


def training_runs() -> dict:
    runs = {a: sum(c.values()) for a, c in DESIGN.items()}
    for sens in SENSITIVITIES.values():
        for a, n in sens.items():
            runs[a] += n
    return runs


def _s_per_iteration(timing: dict, arm: str, h: int) -> float:
    t = timing["train"]
    s1, s2 = t[f"{arm}|h1|t1"]["wall_s_per_iteration"], t[f"{arm}|h2|t1"]["wall_s_per_iteration"]
    return s1 + (s2 - s1) * (h - 1)                      # a + b h through (1, s1) and (2, s2)


def run_hours(timing: dict, arm: str, iterations: int, h: int, c_plus_l_hours: float | None = None) -> float:
    if arm == "C+L":
        if c_plus_l_hours is None:
            raise ValueError("C+L was not timed: pass c_plus_l_hours")
        return c_plus_l_hours
    return _s_per_iteration(timing, "S" if arm == "U-long" else arm, h) * iterations / 3600


def training_budget(timing: dict, iterations: int, h: int, c_plus_l_hours: float = 1.0) -> dict:
    runs = training_runs()
    hours = {a: run_hours(timing, a, iterations, h, c_plus_l_hours) for a in runs}
    per_arm = {a: runs[a] * hours[a] for a in runs}
    lc = DESIGN["U"]["A_lc"] // 9                         # one seed at each of sizes 16 and 128
    reductions = [
        ("degraded-data sensitivity", sum(n * hours[a] for a, n in SENSITIVITIES["degraded"].items())),
        ("U-long", per_arm["U-long"]),
        ("S-Pi", per_arm["S-Pi"]),
        ("latent-dimension sensitivity", sum(n * hours[a] for a, n in SENSITIVITIES["latent_dim"].items())),
        ("MUSCL-C", 0.0),                                 # no training; its evaluation is in the C block
        ("learning-curve sizes 16 and 128 to 2 seeds", sum(2 * lc * hours[a] for a in ("U", "P", "S"))),
        ("System B S-L", DESIGN["S-L"]["B"] * hours["S-L"]),
    ]
    return {"iterations": iterations, "h": h, "runs": runs, "hours_per_run": hours, "per_arm": per_arm,
            "total": sum(per_arm.values()), "unmeasured": ["C+L"],
            "reductions": [{"step": s, "hours": v} for s, v in reductions]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", default=os.path.join(ROOT, "results", "timing_arms.json"))
    ap.add_argument("--iterations", type=int, default=8000)
    ap.add_argument("--h", type=int, default=2)
    ap.add_argument("--c-plus-l-hours", type=float, default=1.0)
    a = ap.parse_args(argv)
    with open(a.timing) as f:
        timing = json.load(f)
    b = training_budget(timing, a.iterations, a.h, a.c_plus_l_hours)
    print(f"training at {a.iterations} iterations, h = {a.h}, one thread per run (CPU-hours):")
    for arm, v in b["per_arm"].items():
        print(f"  {arm:7s} {b['runs'][arm]:3d} runs x {b['hours_per_run'][arm]:6.2f} h = {v:7.1f} h")
    print(f"  total {b['total']:.0f} h (C+L not measured: {a.c_plus_l_hours} h per run assumed)")
    print("reduction order, hours saved:")
    for r in b["reductions"]:
        print(f"  {r['step']}: {r['hours']:.1f}")
    return b


if __name__ == "__main__":
    main()
