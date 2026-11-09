"""Balance and entropy violations (PREREG Section 5; formulation 5.4; Plan 3, Task 2).

Per step of an arm: r_N = N^_{k+1} - N^_k - Q[P_N], r_E likewise with P_E, and r_S = S^_{k+1} -
S^_k - Q[Phi_S] (negative r_S / h is negative total production), with Q the RK4 weights on the
arm's own four stages (one discrete balance definition for every arm). Arms without RK4 stages
(FNO, C+L) use the trapezoid of their own ports at the step's two snapshots, stated with the result.

Normalisation per trajectory-cycle: water by the cycle's feed water, v c_in (t_A + t_D); energy
by |dH| W_ref + integral |wall heat_ref| dt (system B: Q_ref + integral |wall loss| dt).

The reference floor is the same residual on stored data with the ports computed as the models
compute them (decoded-style coarse channels, the stored outlet channel, the coarse wall term),
by the trapezoid over each snapshot interval because stored data have no RK4 stages; the
threshold is 10 x its 99.9th percentile (pilot data).
"""
from __future__ import annotations

import numpy as np
import torch

from p3b.latent.integrate import stage_quadrature

FACTOR = 10.0
PCT = 99.9


def step_residuals(lf, z_prev, z_new, stages, h, raw) -> dict:
    vals = [lf.evaluate(s, raw, grads=False) for s in stages]
    v0, v1 = vals[0], lf.evaluate(z_new, raw, grads=False)
    names = {"E": "P_E", "S": "Phi_S"}
    if lf.system == "A":
        names["N"] = "P_N"
    return {n: v1[n] - v0[n] - stage_quadrature([v[p] for v in vals], h) for n, p in names.items()}


def normalisers(ds) -> dict:
    """{"feed_water" (A only), "energy_scale"}: per trajectory-cycle tensors (traj, cycles)."""
    n_traj, n_cyc = ds.W_ref.shape
    out = {}
    if ds.system == "A":
        feed = []
        for m in ds.meta:
            (_, t_a, inp), (_, t_d, _) = m["schedule"]
            feed.append(m["point"]["v"] * inp[0] * (t_a + t_d))
        out["feed_water"] = torch.tensor(feed, dtype=torch.float64)[:, None].expand(n_traj, n_cyc).clone()
        dH = torch.tensor([abs(_dH(m)) for m in ds.meta], dtype=torch.float64)[:, None]
        out["energy_scale"] = dH * ds.W_ref.abs() + ds.wall_abs
    else:
        out["energy_scale"] = ds.W_ref.abs() + ds.wall_abs
    return out


def _dH(meta) -> float:
    return float(meta["constants"]["DH_C"][meta["material"]])


def trapezoid_residuals(X, raw, t, phys, system: str, n_cells: int) -> dict:
    """Per snapshot interval of one trajectory (flattened snapshots X (T, n_state), raw inputs
    (T, k), times (T,)): r = Q(i+1) - Q(i) - trapezoid of the model-style ports, for E, N (A) and S
    (Phi_S), each interval with its own inputs at both ends. Used for the reference floor and for
    arms without RK4 stages (FNO)."""
    A = system == "A"
    n_ch = 3 if A else 2
    ch = X[:, :n_cells * n_ch].reshape(-1, n_cells, n_ch)
    out = X[:, n_cells * n_ch:]
    left = phys.ports(ch[:-1], out[:-1], raw[:-1])
    right = phys.ports(ch[1:], out[1:], raw[:-1])
    dt = t[1:] - t[:-1]
    idx = {"N": 0, "E": 1, "S": 2} if A else {"E": 0, "S": 1}
    vals = {"E": phys.E(ch), "S": phys.S(ch)}
    if A:
        vals["N"] = phys.N(ch)
    return {q: vals[q][1:] - vals[q][:-1] - 0.5 * (left[i] + right[i]) * dt for q, i in idx.items()}


def reference_floor(ds, phys, pct: float = PCT) -> dict:
    """99.9th percentiles of the normalised model-style residuals of the stored data: by the
    trapezoid of model-style ports ("N", "E"), and against the exact cumulative ports stored with
    the reference ("N_exact", "E_exact"), so the quadrature share of the floor is known."""
    import numpy as _np
    A = ds.system == "A"
    norm = normalisers(ds)
    ns = ds.X.shape[2]
    acc = {k: [] for k in ("N", "E", "N_exact", "E_exact")}
    for j in range(ds.X.shape[0]):
        X = ds.X[j].reshape(-1, ds.X.shape[-1])
        raw = ds.raw_inputs[j].reshape(X.shape[0], -1)
        t = ds.t[j].reshape(-1)
        cyc = torch.arange(X.shape[0] - 1) // ns
        r = trapezoid_residuals(X, raw, t, phys, ds.system, ds.n_cells)
        acc["E"].append((r["E"] / norm["energy_scale"][j, cyc]).abs())
        if A:
            acc["N"].append((r["N"] / norm["feed_water"][j, cyc]).abs())
        P = torch.tensor(_np.load(ds.paths[j])["ports_cum"], dtype=torch.float64).reshape(X.shape[0], -1)
        n_ch = 3 if A else 2
        ch = X[:, :ds.n_cells * n_ch].reshape(-1, ds.n_cells, n_ch)
        E = phys.E(ch)
        if A:
            net_E = P[:, 2] - P[:, 3] - P[:, 4]
            N = phys.N(ch)
            net_N = P[:, 0] - P[:, 1]
            acc["N_exact"].append(((N[1:] - N[:-1]) - (net_N[1:] - net_N[:-1])).abs() / norm["feed_water"][j, cyc])
        else:
            net_E = P[:, 0] - P[:, 1] - P[:, 2]
        acc["E_exact"].append(((E[1:] - E[:-1]) - (net_E[1:] - net_E[:-1])).abs() / norm["energy_scale"][j, cyc])
    return {k: float(np.percentile(torch.cat(v).numpy(), pct)) for k, v in acc.items() if v}


def count_violations(r, scale, floor: float, factor: float = FACTOR) -> int:
    """Steps whose normalised residual |r| / scale exceeds factor x floor."""
    return int(((r.abs() / scale) > factor * floor).sum())
