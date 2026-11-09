"""Evaluation of one trained arm on one test set, and the results file (Plan 3, Task 8).

Per trajectory (and the seed it was trained with): e_k per cycle (physical-field nRMSE), epsilon_k
on W_k (A) or Q_k (B) with the same quadrature for model and reference, the failure cycle with
capping at 0.5, the violation counts (water, energy: normalised residual above 10 x the reference
floor; entropy: production per second below -tau_S), the projected arms' defect flags, and CPU
time. The floors and tau_S are pilot values (PREREG 10), passed in.
"""
from __future__ import annotations

import json
import os

import torch

from p3b._paper1 import PAPER1_SHA256
from p3b.eval.metrics import apply_failure, cycle_errors, relative_error, release
from p3b.eval.rollout import rollout
from p3b.eval.violations import FACTOR, normalisers
from p3b.provenance import git_commit


def evaluate(arm, ds, phys, n_cycles: int, h_steps: int, floors: dict, tau_S: float, seed: int) -> list:
    A = ds.system == "A"
    n_out = 2 if A else 1
    rhoc = None if A else phys.rhoc
    norm = normalisers(ds)
    ns = ds.X.shape[2]
    rows = []
    for j in range(ds.X.shape[0]):
        r = rollout(arm, ds, j, n_cycles, h_steps, track=arm.lf is not None)
        ref = ds.X[j, :n_cycles]
        pred = torch.where(torch.isfinite(r.X), r.X, ref)              # failed snapshots: capped below
        e = cycle_errors(pred, ref, phys, ds.n_cells)
        raw, t = ds.raw_inputs[j, :n_cycles], ds.t[j, :n_cycles]
        W_p = release(pred[..., -n_out:], raw, t, ds.system, rhoc)
        W_r = release(ref[..., -n_out:], raw, t, ds.system, rhoc)
        e, eps, failed = apply_failure(e, relative_error(W_p, W_r), r.finite)
        viol = {"N": 0, "E": 0, "S": 0}
        if r.residuals:
            n_int = len(r.residuals["E"])
            cyc = torch.arange(n_int) // ns
            dt = (t.reshape(-1)[1:] - t.reshape(-1)[:-1])
            dt = torch.cat([dt, dt[-1:]])[:n_int]
            if "N" in r.residuals and A:
                viol["N"] = int(((r.residuals["N"] / norm["feed_water"][j, cyc]).abs() > FACTOR * floors["N"]).sum())
            viol["E"] = int(((r.residuals["E"] / norm["energy_scale"][j, cyc]).abs() > FACTOR * floors["E"]).sum())
            viol["S"] = int(((r.residuals["S"] / dt) < -tau_S).sum())
        rows.append({"trajectory": j, "seed": int(seed), "e": [float(x) for x in e],
                     "eps": [float(x) for x in eps], "failed_cycle": failed, "violations": viol,
                     "flags": {k: int(v) for k, v in r.flags.items()}, "cpu_s": float(r.cpu_s)})
    return rows


def write_results(path, rows: list, meta: dict) -> None:
    """Rows plus metadata (commit, Paper 1 hashes, and whatever the caller adds: arm, system,
    material, parameter counts, pilot values), written atomically."""
    meta = dict(meta, commit=meta.get("commit", git_commit()), paper1_sha256=dict(PAPER1_SHA256))
    path = os.fspath(path)
    tmp = path + ".partial"
    with open(tmp, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=1)
    os.replace(tmp, path)
