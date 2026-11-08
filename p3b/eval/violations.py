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


def reference_floor(ds, phys, pct: float = PCT) -> dict:
    """99.9th percentile of the normalised model-style residuals of the stored data."""
    A = ds.system == "A"
    n_ch, nc = (3 if A else 2), ds.n_cells
    norm = normalisers(ds)
    rN, rE = [], []
    for j in range(ds.X.shape[0]):
        X = ds.X[j].reshape(-1, ds.X.shape[-1])
        raw = ds.raw_inputs[j].reshape(X.shape[0], -1)
        t = ds.t[j].reshape(-1)
        ch = X[:, :nc * n_ch].reshape(-1, nc, n_ch)
        out = X[:, nc * n_ch:]
        E = phys.E(ch)
        ports = phys.ports(ch, out, raw)
        P_E = ports[1] if A else ports[0]
        ns = ds.X.shape[2]
        cyc = torch.arange(X.shape[0] - 1) // ns
        dt = t[1:] - t[:-1]
        # the interval's own inputs at both ends (inputs are constant over an interval)
        P_E_right = (phys.ports(ch[1:], out[1:], raw[:-1])[1 if A else 0])
        rE.append(((E[1:] - E[:-1] - 0.5 * (P_E[:-1] + P_E_right) * dt) / norm["energy_scale"][j, cyc]).abs())
        if A:
            N = phys.N(ch)
            P_N, P_N_right = ports[0], phys.ports(ch[1:], out[1:], raw[:-1])[0]
            rN.append(((N[1:] - N[:-1] - 0.5 * (P_N[:-1] + P_N_right) * dt) / norm["feed_water"][j, cyc]).abs())
    res = {"E": float(np.percentile(torch.cat(rE).numpy(), pct))}
    if A:
        res["N"] = float(np.percentile(torch.cat(rN).numpy(), pct))
    return res


def count_violations(r, scale, floor: float, factor: float = FACTOR) -> int:
    """Steps whose normalised residual |r| / scale exceeds factor x floor."""
    return int(((r.abs() / scale) > factor * floor).sum())
