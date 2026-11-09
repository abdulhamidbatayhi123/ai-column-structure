"""Test whether the Task 10 stall (MOF-303, box centre, 1024 cells, cycle 1) comes from the
finite-difference Jacobian: integrate the first adsorption half with scipy's BDF, (a) with the
finite-difference Jacobian on the sparsity pattern (the reference default) and (b) with the exact
Jacobian from the torch right-hand side (p3b.solvers.c_plus_l.TorchColumnRHS, which matches
Column.rhs to 1e-12) by 9-colour forward-mode products. Both with a CPU cap; prints the time reached,
evaluations, Jacobians and CPU.

    python verify/stall_exact_jacobian.py --nz 1024 --cap 600 --jac exact|fd
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from p3b.simulate_a import ATOL, RTOL, prewet_state, setup_a  # noqa: E402
from p3b.solvers.c_plus_l import TorchColumnRHS, colored_jacobian  # noqa: E402

CENTRE = {"RH_ads": 0.4, "T_ads": 300.5, "T_des": 348.0, "v": 0.165, "alpha": 0.85, "beta": 0.75,
          "index": 0, "n_cycles": 2}


class Cap(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nz", type=int, default=1024)
    ap.add_argument("--cap", type=float, default=600.0)
    ap.add_argument("--jac", default="exact")
    a = ap.parse_args()
    torch.set_num_threads(1)
    col, sch = setup_a(CENTRE, "MOF-303", a.nz)
    _, dur, inputs = sch[0]
    y0 = prewet_state(col, sch[0][2][0], CENTRE["T_ads"])
    trhs = TorchColumnRHS(col)
    inp = torch.tensor(inputs, dtype=torch.float64)
    cnt = {"f": 0, "J": 0, "t": 0.0}
    t0 = time.process_time()

    def f(t, y):
        cnt["f"] += 1
        cnt["t"] = max(cnt["t"], t)
        if time.process_time() - t0 > a.cap:
            raise Cap()
        return col.rhs(t, y, inputs)

    def jac(t, y):
        cnt["J"] += 1
        J = colored_jacobian(lambda x: trhs(x, inp), torch.tensor(y, dtype=torch.float64), a.nz, 3)
        return csr_matrix(J.numpy())

    kw = dict(jac=jac) if a.jac == "exact" else dict(jac_sparsity=col.sparsity())
    try:
        sol = solve_ivp(f, (0.0, dur), y0, method="BDF", rtol=RTOL, atol=ATOL, **kw)
        status = sol.message
    except Cap:
        status = "CPU cap reached"
    print(json.dumps({"nz": a.nz, "jac": a.jac, "dur": dur, "t_reached": cnt["t"], "nfev": cnt["f"],
                      "njev_exact": cnt["J"], "cpu_s": time.process_time() - t0, "status": status}), flush=True)


if __name__ == "__main__":
    main()
