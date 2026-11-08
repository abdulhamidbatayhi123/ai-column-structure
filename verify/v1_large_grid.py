"""V1 at a large grid: with c_w = 0 and Paper 1's physics, the consistent column reproduces Paper 1's
solver_fd.py (formulation Section 3.4). Same material, condition, horizon and tolerances as
verify/run_checks.py V1 (N = 200 there); here N is a parameter (plan 1 Task 10: N = 2048).

    python verify/v1_large_grid.py --n 2048        -> verify/results_v1_2048.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from p3b import _paper1  # noqa: E402,F401
from p3b.column import Column, from_paper1  # noqa: E402
from p3b.provenance import git_commit  # noqa: E402
from gen_parametric_dataset import build_physics, rh_to_conc  # noqa: E402  (Paper 1, read-only)
from solver_fd import generate_breakthrough_data  # noqa: E402  (Paper 1, read-only)

MAT = dict(q_max=24.5, delta_H=-52000.0, step_rh=0.12, isotherm_n=3.0,
           henry_fraction=0.17, d_p=3.25e-3, rho_p=1050.0, eps_t=0.39)
COND = dict(T_in=303.15, rh_feed=0.40, v=0.10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    N = a.n
    t0 = time.process_time()
    phys = build_physics(MAT, COND)
    c_in = rh_to_conc(COND["rh_feed"], COND["T_in"])
    t_end = 1.0 * phys.stoichiometric_time(c_in, COND["T_in"])
    _, _, yp = generate_breakthrough_data(phys, N_z=N, t_final=t_end, c_in=c_in, n_snapshots=41,
                                          rtol=1e-9, atol=1e-13, verbose=False)
    col = Column(phys, N, c_w=0.0, iso=from_paper1(phys))
    _, ym = col.integrate(col.initial_state(phys.T_w), (0.0, t_end), (c_in, phys.T_in, phys.T_w), 41,
                          rtol=1e-9, atol=1e-13)
    n, ad, u = col.split(ym)
    c_m, q_m, T_m = n / col.eps, ad / col.A, col.temperature(n, ad, u)
    c_p, q_p, T_p = yp[:N], yp[N:2 * N], yp[2 * N:]
    out = {"N": N, "horizon_t_st": 1.0, "material": MAT, "condition": COND, "commit": git_commit(),
           "max_rel_diff_c": float(np.max(np.abs(c_m - c_p)) / c_in),
           "max_rel_diff_q": float(np.max(np.abs(q_m - q_p)) / phys.q_max),
           "max_abs_diff_T_K": float(np.max(np.abs(T_m - T_p))),
           "T_rise_K": float(T_p.max() - phys.T_w), "cpu_s": round(time.process_time() - t0, 1),
           "bounds": {"c": 2e-9, "q": 1e-8, "T_K": 2e-7}}
    path = a.out or os.path.join(ROOT, "verify", f"results_v1_{N}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
