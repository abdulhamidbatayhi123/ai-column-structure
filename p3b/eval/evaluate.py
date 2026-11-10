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
from p3b.eval import rollout as ro
from p3b.eval.violations import FACTOR, normalisers, trapezoid_residuals
from p3b.latent.metrics import field_nrmse
from p3b.provenance import git_commit


def evaluate(arm, ds, phys, n_cycles: int, h_steps: int, floors: dict, tau_S: float, seed: int) -> list:
    """One row per test trajectory. Violations: latent arms by their own RK4 stages (they need
    LatentFunctionals, else ValueError); the FNO by the trapezoid on its own snapshots
    (violations_method says which); None where not computed, never a silent 0."""
    A = ds.system == "A"
    if arm.kind != "FNO" and arm.lf is None:
        raise ValueError("a latent arm needs its LatentFunctionals (lf) for the violation counts")
    n_out, n_ch = (2, 3) if A else (1, 2)
    rhoc = None if A else phys.rhoc
    norm = normalisers(ds)
    ns = ds.X.shape[2]
    rows = []
    results = ro.rollouts(arm, ds, n_cycles, h_steps, track=arm.lf is not None)   # batched (PREREG 5)
    for j, r in enumerate(results):
        ref = ds.X[j, :n_cycles]
        pred = torch.where(torch.isfinite(r.X), r.X, ref)              # failed snapshots: capped below
        e = cycle_errors(pred, ref, phys, ds.n_cells)
        nc = ds.n_cells
        e_fields = []
        for k in range(n_cycles):
            P_ = phys.physical_fields(pred[k][:, :nc * n_ch].reshape(-1, nc, n_ch)).reshape(ns, -1)
            T_ = phys.physical_fields(ref[k][:, :nc * n_ch].reshape(-1, nc, n_ch)).reshape(ns, -1)
            e_fields.append([float(v) for v in field_nrmse(P_, T_, nc, n_ch, per_field=True)])
        raw, t = ds.raw_inputs[j, :n_cycles], ds.t[j, :n_cycles]
        W_p = release(pred[..., -n_out:], raw, t, ds.system, rhoc)
        W_r = release(ref[..., -n_out:], raw, t, ds.system, rhoc)
        eps_raw = relative_error(W_p, W_r)
        e_c, eps_c, failed = apply_failure(e, eps_raw, r.finite)
        viol, method = {"N": None, "E": None, "S": None}, None
        res = r.residuals
        t_flat = t.reshape(-1)
        if arm.kind == "FNO" and r.failed_cycle is None:
            X_flat = pred.reshape(-1, pred.shape[-1])
            res = trapezoid_residuals(X_flat, raw.reshape(X_flat.shape[0], -1), t_flat, phys, ds.system, nc)
            method = "trapezoid on own snapshots"
        elif res:
            method = "RK4 stages"
        if res:
            n_int = len(res["E"])
            cyc = torch.arange(n_int) // ns
            dt = torch.cat([t_flat[1:] - t_flat[:-1], (t_flat[-1:] - t_flat[-2:-1])])[:n_int]
            viol = {"N": None, "E": None, "S": None}
            if A and "N" in res:
                viol["N"] = int(((res["N"] / norm["feed_water"][j, cyc]).abs() > FACTOR * floors["N"]).sum())
            viol["E"] = int(((res["E"] / norm["energy_scale"][j, cyc]).abs() > FACTOR * floors["E"]).sum())
            viol["S"] = int(((res["S"] / dt) < -tau_S).sum())
        rows.append({"trajectory": j, "seed": int(seed), "e": [float(x) for x in e_c],
                     "eps": [float(x) for x in eps_c], "failed_cycle": failed, "violations": viol,
                     "violations_method": method, "flags": {k: int(v) for k, v in r.flags.items()},
                     "cpu_s": float(r.cpu_s), "e_fields": e_fields, "failure_reason": r.error,
                     "e_uncapped": [float(x) for x in e], "eps_uncapped": [float(x) for x in eps_raw],
                     "W_pred": [float(x) for x in W_p], "W_ref": [float(x) for x in W_r],
                     "W_ref_exact": [float(x) for x in ds.W_ref[j, :n_cycles]]})
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


def evaluate_files(files_by_arm: dict, ds, phys, n_cycles: int, seed: int, floors: dict | None = None,
                   tau_S: float | None = None) -> list:
    """Score stored solver-arm files (C, MUSCL-C, C+L; one file per test trajectory, in the order of
    ds) against the reference with the same metrics as `evaluate`; violations by the trapezoid on
    the arm's own snapshots, counted when the pilot floors and tau_S are given (None otherwise); the
    normalised residuals are kept either way."""
    from p3b.latent.data import load_set
    A = ds.system == "A"
    n_out, n_ch = (2, 3) if A else (1, 2)
    rhoc = None if A else phys.rhoc
    norm = normalisers(ds)
    ns, nc = ds.X.shape[2], ds.n_cells
    rows = []
    for arm_name, paths in files_by_arm.items():
        for j, p in enumerate(paths):
            if p is None:                                 # the solver failed: a failure at cycle 1
                pred_all = torch.full_like(ds.X[j, :n_cycles], float("nan"))
            else:
                pred_all = load_set([p]).X[0, :n_cycles]
            ref = ds.X[j, :n_cycles]
            finite = torch.isfinite(pred_all).reshape(n_cycles, -1).all(-1)
            pred = torch.where(torch.isfinite(pred_all), pred_all, ref)
            e = cycle_errors(pred, ref, phys, nc)
            raw, t = ds.raw_inputs[j, :n_cycles], ds.t[j, :n_cycles]
            W_p = release(pred[..., -n_out:], raw, t, ds.system, rhoc)
            W_r = release(ref[..., -n_out:], raw, t, ds.system, rhoc)
            eps_raw = relative_error(W_p, W_r)
            e_c, eps_c, failed = apply_failure(e, eps_raw, finite)
            X_flat = pred.reshape(-1, pred.shape[-1])
            t_flat = t.reshape(-1)
            res = trapezoid_residuals(X_flat, raw.reshape(X_flat.shape[0], -1), t_flat, phys, ds.system, nc)
            cyc = torch.arange(len(res["E"])) // ns
            dt = t_flat[1:] - t_flat[:-1]
            viol = {"N": None, "E": None, "S": None}
            if floors is not None:
                if A:
                    viol["N"] = int(((res["N"] / norm["feed_water"][j, cyc]).abs() > FACTOR * floors["N"]).sum())
                viol["E"] = int(((res["E"] / norm["energy_scale"][j, cyc]).abs() > FACTOR * floors["E"]).sum())
                viol["S"] = int(((res["S"] / dt) < -tau_S).sum())
            rows.append({"arm": arm_name, "trajectory": j, "seed": int(seed), "e": [float(x) for x in e_c],
                         "eps": [float(x) for x in eps_c], "failed_cycle": failed,
                         "violations": viol, "violations_method": "trapezoid on own snapshots",
                         "residuals_normalised": {
                             "E": [float(x) for x in (res["E"] / norm["energy_scale"][j, cyc])],
                             **({"N": [float(x) for x in (res["N"] / norm["feed_water"][j, cyc])]} if A else {}),
                             "S_per_s": [float(x) for x in (res["S"] / dt)]},
                         "e_uncapped": [float(x) for x in e], "eps_uncapped": [float(x) for x in eps_raw],
                         "W_pred": [float(x) for x in W_p], "W_ref": [float(x) for x in W_r],
                         "W_ref_exact": [float(x) for x in ds.W_ref[j, :n_cycles]]})
    return rows
