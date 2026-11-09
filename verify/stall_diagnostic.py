"""Diagnose the BDF stall seen on Kaggle (Task 10): MOF-303 at the box centre, cycle 1, 1024 cells,
took 3,753 CPU-s and 642,420 function evaluations (512 cells: 9 s, 2,338), with scipy num_jac overflow
warnings. This script integrates the first adsorption half with the reference settings and logs
progress (time reached, evaluations, CPU) and warnings, with a CPU cap, for the augmented system
(integral states included) and for the physical system alone (no integral states).

    python verify/stall_diagnostic.py [--nz 1024] [--cap 900] [--variant augmented|physical]
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from scipy.integrate import solve_ivp  # noqa: E402

from p3b.simulate_a import ATOL, RTOL, _Augmented, prewet_state, setup_a  # noqa: E402

CENTRE = {"RH_ads": 0.4, "T_ads": 300.5, "T_des": 348.0, "v": 0.165, "alpha": 0.85, "beta": 0.75,
          "index": 0, "n_cycles": 2}


class Cap(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nz", type=int, default=1024)
    ap.add_argument("--cap", type=float, default=900.0)
    ap.add_argument("--variant", default="augmented")
    a = ap.parse_args()
    col, schedule = setup_a(CENTRE, "MOF-303", a.nz)
    c_in = schedule[0][2][0]
    y_phys = prewet_state(col, c_in, CENTRE["T_ads"])
    _, dur, inputs = schedule[0]
    aug = _Augmented(col)
    if a.variant == "augmented":
        y0 = np.concatenate([y_phys, np.zeros(a.nz + 6)])
        f, sp = aug.rhs, aug.sparsity
    else:
        y0 = y_phys
        f, sp = (lambda t, y, inp: col.rhs(t, y, inp)), col.sparsity()
    calls = {"n": 0, "t": 0.0}
    t0 = time.process_time()
    log = []

    def g(t, y, inp):
        calls["n"] += 1
        calls["t"] = max(calls["t"], t)
        if calls["n"] % 20000 == 0:
            log.append({"nfev": calls["n"], "t": calls["t"], "cpu": time.process_time() - t0})
            print(json.dumps(log[-1]), flush=True)
        if time.process_time() - t0 > a.cap:
            raise Cap()
        return f(t, y, inp)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            sol = solve_ivp(g, (0.0, dur), y0, method="BDF", args=(inputs,), jac_sparsity=sp, rtol=RTOL, atol=ATOL)
            status = sol.message
        except Cap:
            status = "CPU cap reached"
        warn = [(str(x.message)[:80], x.filename.split("/")[-1].split("\\")[-1], x.lineno) for x in w]
    out = {"nz": a.nz, "variant": a.variant, "dur": dur, "t_reached": calls["t"], "nfev": calls["n"],
           "cpu_s": time.process_time() - t0, "status": status, "n_warnings": len(warn), "first_warnings": warn[:5]}
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
