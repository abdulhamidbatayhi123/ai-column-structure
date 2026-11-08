"""Reference gates on stored trajectories (PREREG_P3B_draft.md, Section 5.3; plan 1 Global Constraints).

All gates read only the coarse npz written by p3b.store.Coarsener (via p3b.generate):
  * stored_closure: water (A) and energy against the stored cumulative ports, <= 1e-5 relative;
  * coarse_entropy_margin: per snapshot interval, the change of the coarse entropy (entropy of
    the block averages) minus the port entropy flux integrated over the interval (inlet stream
    exact, outlet stream by the trapezoid rule on the stored outlet, wall heat over the bath
    temperature exactly), divided by the sum of the magnitudes of those terms; the minimum must
    be positive;
  * floor_share: fine cell-snapshots below the concentration/loading floor, < 0.01;
  * css_cycle: the first cycle k >= 2 (1-based) whose release W_k satisfies
    |W_k - W_(k-1)| / |W_k| < CSS_TOL. W is water released in the desorption half (A) or
    energy released in the discharge half (B), net of the inlet, over the whole half: from
    snapshot n_snap/2 to the next cycle's snapshot 0 (the stored `ports_end` for the last cycle).
    This is PREREG Section 5's W_k (A) and Q_k (B) for the reference.
"""
from __future__ import annotations

import json
import os

import numpy as np

CSS_TOL = 1e-3
CLOSURE_TOL = 1e-5
FLOOR_SHARE_TOL = 0.01


def _load(npz):
    if isinstance(npz, (str, os.PathLike)):
        return np.load(npz, allow_pickle=False)
    return npz


def load_meta(npz) -> dict:
    return json.loads(str(_load(npz)["meta"]))


def model_from_meta(meta: dict):
    """The fine model the trajectory was generated with (Column for A, ThermalStore for B)."""
    if meta["system"] == "A":
        from p3b.simulate_a import setup_a
        return setup_a(meta["point"], meta["material"], meta["N_z"])[0]
    from p3b.simulate_b import make_store
    return make_store(meta["N_z"])


def _flat(a):
    """(n_cycles, n_snap, ...) -> (n_cycles * n_snap, ...)."""
    return a.reshape((-1,) + a.shape[2:])


def stored_closure(npz) -> float:
    d = _load(npz)
    meta = load_meta(d)
    dz = meta["L"] / meta["N_c"]
    cons, P = _flat(d["conserved_blocks"]), _flat(d["ports_cum"])
    errs = []
    if meta["system"] == "A":
        content = [(cons[..., 0] + cons[..., 1]).sum(axis=1) * dz, cons[..., 2].sum(axis=1) * dz]
        net = [P[:, 0] - P[:, 1], P[:, 2] - P[:, 3] - P[:, 4]]
        scale = [np.abs(P[:, :2]).max(), np.abs(P[:, 2:5]).max()]
    else:
        content = [(cons[..., 0] + cons[..., 1]).sum(axis=1) * dz]
        net = [P[:, 0] - P[:, 1] - P[:, 2]]
        scale = [np.abs(P[:, :3]).max()]
    for c, n, s in zip(content, net, scale):
        errs.append(np.max(np.abs((c - c[0]) - (n - n[0]))) / s)
    return float(max(errs))


def coarse_entropy_margin(npz, model) -> float:
    d = _load(npz)
    meta = load_meta(d)
    dz = meta["L"] / meta["N_c"]
    ns = d["t"].shape[1]
    t, cons, P, out = _flat(d["t"]), _flat(d["conserved_blocks"]), _flat(d["ports_cum"]), \
        _flat(d["outlet"]).astype(float)
    if meta["system"] == "A":
        S = model.cell_entropy(cons[..., 0], cons[..., 1], cons[..., 2]).sum(axis=1) * dz
        v = model.p.v
        s_out = v * model.stream_entropy(np.exp(out[:, 0]), np.exp(out[:, 1]))
        wall = P[:, 4]

        def s_in(inp):
            return v * model.stream_entropy(inp[0], inp[1])

        def s_out_pair(i, inp):
            return s_out[i], s_out[i + 1]
    else:
        from p3b.thermal_store import T0
        Tf, Ts = T0 + cons[..., 0] / model.Cf, T0 + cons[..., 1] / model.Cs
        S = (model.Cf * np.log(Tf / T0) + model.Cs * np.log(Ts / T0)).sum(axis=1) * dz
        ends = _flat(d["ends"])                      # T_f at z = 0 and z = L
        wall = P[:, 2]

        def s_in(inp):
            return abs(inp[0]) * model.stream_entropy(inp[1])

        def s_out_pair(i, inp):   # outlet of this interval's direction, at both of its ends
            e = 1 if inp[0] > 0 else 0
            return abs(inp[0]) * model.stream_entropy(ends[i, e]), abs(inp[0]) * model.stream_entropy(ends[i + 1, e])
    schedule = meta["schedule"]
    margins = []
    for i in range(len(t) - 1):
        inp = schedule[0 if (i % ns) < ns // 2 else 1][2]
        T_bath = inp[2]
        dt = t[i + 1] - t[i]
        o0, o1 = s_out_pair(i, inp)
        terms = (S[i + 1] - S[i], s_in(inp) * dt, 0.5 * (o0 + o1) * dt, (wall[i + 1] - wall[i]) / T_bath)
        prod = terms[0] - (terms[1] - terms[2] - terms[3])
        margins.append(prod / sum(abs(x) for x in terms))
    return float(min(margins))


def floor_share(npz) -> float:
    d = _load(npz)
    return float(d["floored_cells"]) / float(d["checked_cells"])


def cycle_release(npz) -> np.ndarray:
    d = _load(npz)
    P = d["ports_cum"]
    ns = P.shape[1]
    net_out = P[..., 1] - P[..., 0]      # A: water out - in; B: energy out - in (columns 0, 1 in both)
    end = d["ports_end"]
    cycle_end = np.append(net_out[1:, 0], end[1] - end[0])   # next cycle's start; last: end of run
    return cycle_end - net_out[:, ns // 2]


def css_cycle(x) -> int | None:
    """First 1-based cycle k >= 2 with |W_k - W_(k-1)| / |W_k| < CSS_TOL; x is an npz or W."""
    W = cycle_release(x) if isinstance(x, (str, os.PathLike, np.lib.npyio.NpzFile)) else np.asarray(x, float)
    for k in range(1, len(W)):
        if abs(W[k] - W[k - 1]) / abs(W[k]) < CSS_TOL:
            return k + 1
    return None
