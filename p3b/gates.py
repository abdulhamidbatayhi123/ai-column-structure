"""Reference gates on stored trajectories (PREREG_P3B_draft.md, Section 5.3; plan 1 Global Constraints).

All gates read only the coarse npz written by p3b.store.Coarsener (via p3b.generate):
  * stored_closure: water (A) and energy against the stored cumulative ports, <= 1e-5 relative;
  * entropy_gate: per snapshot interval, the change of the coarse entropy (entropy of the block
    averages) minus the port entropy flux over the interval, from the exact cumulative entropy
    ports; min rate >= -tau_ref (see entropy_gate);
  * rate_identity_max, fine_production_min: the fine-level quantities computed at every snapshot
    before the fine state was discarded (rate identities <= 1e-12; production >= 0 above the floor);
  * closure_ok: the integrated closure with cumulative-flux states, <= 10 x rtol;
  * production_ready: not a smoke output (no cycle override, the design's cycle count);
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


def _entropy_intervals(npz, model) -> dict:
    """Per snapshot interval: the port entropy flux (exact: stream entropy from the cumulative entropy
    ports, wall heat over the bath temperature, constant within a half-cycle), the change of the
    coarse entropy (entropy of the block averages, blocks floored as in the Global Constraints) and,
    where stored, of the fine entropy; the turnover is the sum of the magnitudes of the terms."""
    d = _load(npz)
    meta = load_meta(d)
    dz = meta["L"] / meta["N_c"]
    ns = d["t"].shape[1]
    t, cons, P = _flat(d["t"]), _flat(d["conserved_blocks"]), _flat(d["ports_cum"])
    floored = 0
    if meta["system"] == "A":
        from p3b.store import FLOOR
        n_min = model.eps * FLOOR * meta["schedule"][0][2][0]
        a_min = model.A * FLOOR * model.iso.q_capacity
        floored = int(np.sum((cons[..., 0] < n_min) | (cons[..., 1] < a_min)))
        with np.errstate(all="ignore"):
            S = model.cell_entropy(np.maximum(cons[..., 0], n_min), np.maximum(cons[..., 1], a_min),
                                   cons[..., 2]).sum(axis=1) * dz
        wall, s_in, s_out = P[:, 4], P[:, 5], P[:, 6]
    else:
        from p3b.thermal_store import T0
        Tf, Ts = T0 + cons[..., 0] / model.Cf, T0 + cons[..., 1] / model.Cs
        S = (model.Cf * np.log(Tf / T0) + model.Cs * np.log(Ts / T0)).sum(axis=1) * dz
        wall, s_in, s_out = P[:, 2], P[:, 4], P[:, 5]
    S_fine = _flat(d["S_fine"]) if "S_fine" in d.files else None
    schedule = meta["schedule"]
    k = np.arange(len(t) - 1)
    T_bath = np.array([schedule[0 if (i % ns) < ns // 2 else 1][2][2] for i in k])
    dt = np.diff(t)
    phi = np.diff(s_in) - np.diff(s_out) - np.diff(wall) / T_bath
    dS = np.diff(S)
    turnover = np.abs(dS) + np.abs(np.diff(s_in)) + np.abs(np.diff(s_out)) + np.abs(np.diff(wall) / T_bath)
    out = {"dt": dt, "coarse_rate": (dS - phi) / dt, "turnover_rate": turnover / dt,
           "coarse_margin": (dS - phi) / turnover, "floored_blocks": floored}
    out["fine_rate"] = (np.diff(S_fine) - phi) / dt if S_fine is not None else None
    return out


def coarse_entropy_margin(npz, model) -> float:
    """The smallest (dS_coarse - Phi_S) over an interval, relative to the interval's entropy turnover."""
    m = _entropy_intervals(npz, model)["coarse_margin"]
    return float(np.min(m)) if np.all(np.isfinite(m)) else float("nan")


def entropy_gate(npz, model, tau_ref: float | None = None) -> dict:
    """PREREG Section 5.3: min_t (dS_coarse/dt - Phi_S) >= -tau_ref, by exact ports.

    tau_ref is the reference's own numerical noise floor for this statistic: at the fine level the
    production is non-negative by proof, so any negative value of the same statistic computed with
    the fine entropy is numerical noise. tau_ref = 10 x the 99.9th percentile of its negative part,
    floored at 10 x rtol x the median entropy turnover rate. A pilot-wide value can be passed in;
    otherwise it is computed from this trajectory and labelled provisional.
    """
    from p3b.simulate_a import RTOL
    e = _entropy_intervals(npz, model)
    rate, fine = e["coarse_rate"], e["fine_rate"]
    floor_tau = 10.0 * RTOL * float(np.median(e["turnover_rate"]))
    source = "given"
    if tau_ref is None:
        source = "provisional (this trajectory)"
        neg = np.maximum(0.0, -fine[np.isfinite(fine)]) if fine is not None else np.zeros(1)
        tau_ref = max(10.0 * float(np.percentile(neg, 99.9)) if neg.size else 0.0, floor_tau)
    finite = bool(np.all(np.isfinite(rate)))
    min_rate = float(np.min(rate)) if finite else float("nan")
    return {"min_rate": min_rate, "tau_ref": float(tau_ref), "tau_source": source, "passed": finite and
            min_rate >= -tau_ref, "min_rate_fine": float(np.nanmin(fine)) if fine is not None else None,
            "floored_blocks": e["floored_blocks"], "n_intervals": int(rate.size)}


def rate_identity_max(npz) -> float:
    """Largest relative rate-identity residual over all stored snapshots (floored snapshots excluded)."""
    r = _load(npz)["rate_residual"]
    return float(np.nanmax(r)) if np.any(np.isfinite(r)) else float("nan")


def fine_production_min(npz) -> float:
    """Smallest mechanism production on cells above the floor, relative to the production per cell."""
    f = _load(npz)["fine_min_production"]
    return float(np.nanmin(f)) if np.any(np.isfinite(f)) else float("nan")


def closure_ok(npz) -> bool:
    """Integrated closure with cumulative-flux states, <= 10 x rtol, every half-cycle."""
    return float(load_meta(npz)["closure_max"]) <= CLOSURE_TOL


def production_ready(npz) -> tuple[bool, str]:
    """Not a smoke output: no cycle override, the design's cycle count, 200 snapshots per cycle."""
    d = _load(npz)
    meta = load_meta(d)
    if meta.get("cycles_override") is not None:
        return False, f"cycles override {meta['cycles_override']} (smoke output)"
    n_cyc = d["t"].shape[0]
    if meta["point"].get("n_cycles") != n_cyc:
        return False, f"{n_cyc} cycles stored, the design point has {meta['point'].get('n_cycles')}"
    if d["t"].shape[1] != 200:
        return False, f"{d['t'].shape[1]} snapshots per cycle, not 200"
    return True, ""


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
