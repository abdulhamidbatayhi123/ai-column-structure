"""Cycle metrics (PREREG Section 5; Plan 3, Task 1).

e_k: Paper 1's nRMSE per physical field (A: c, q, T; B: T_f, T_s) over cycle k's snapshots and the
coarse cells, averaged over the fields.

W_k (A) = integral over the desorption half of v (c_out - c_in,des) dt, and Q_k (B) = integral
over the discharge half of |v| rho_f c_f (T_out - T_cold) dt, from an arm's own outlet channel at
its snapshots: the trapezoid over the half's snapshots. System A closes the half with the next
cycle's first snapshot (or a given closing state); the outlet is at z = L in both halves. In
system B the outlet channel follows the flow direction, so the next cycle's first snapshot is
already the other end: the last interval of the discharge is taken by the left rectangle. The
reference W used in epsilon comes from the same quadrature on the stored outlet, so a
quadrature error never counts as model error; the exact-port W is reported beside it.

Failure at cycle k (1-based): a non-finite state, epsilon > 0.5 or e_k > 0.5; from then on every
value is capped at 0.5.
"""
from __future__ import annotations

import torch

from p3b.latent.metrics import field_nrmse

CAP = 0.5


def cycle_errors(pred_X, true_X, phys, n_cells: int):
    """(cycles, snaps, n_state) -> e (cycles,): mean over the physical fields of Paper 1's nRMSE."""
    n_ch = 3 if pred_X.shape[-1] == n_cells * 3 + 2 else 2
    out = []
    for k in range(pred_X.shape[0]):
        P = phys.physical_fields(pred_X[k][:, :n_cells * n_ch].reshape(-1, n_cells, n_ch)).reshape(len(pred_X[k]), -1)
        T = phys.physical_fields(true_X[k][:, :n_cells * n_ch].reshape(-1, n_cells, n_ch)).reshape(len(true_X[k]), -1)
        out.append(field_nrmse(P, T, n_cells, n_ch))
    return torch.stack(out)


def release(outlet, raw, t, system: str, rhoc: float | None = None, closing=None):
    """W_k (A) or Q_k (B) per cycle from the outlet channels (cycles, snaps, n_out), the raw inputs
    (cycles, snaps, k) and the times (cycles, snaps). `closing` = (outlet, time) after the last
    snapshot closes the last cycle of system A; without it, A returns one value fewer."""
    nc, ns = t.shape
    half = ns // 2
    out_f = outlet.reshape(nc * ns, -1)
    t_f = t.reshape(nc * ns)
    raw_f = raw.reshape(nc * ns, -1)
    if system == "A":
        f = lambda o, r: r[..., 3] * (torch.exp(o[..., 0]) - r[..., 0])  # noqa: E731
    else:
        f = lambda o, r: r[..., 0].abs() * rhoc * (o[..., 0] - r[..., 1])  # noqa: E731
    vals = []
    for k in range(nc):
        i0, i1 = k * ns + half, (k + 1) * ns            # the half's snapshots are i0 .. i1 - 1
        r = raw_f[i0]
        y = f(out_f[i0:i1], r.expand(i1 - i0, -1))
        tt = t_f[i0:i1]
        W = (0.5 * (y[1:] + y[:-1]) * (tt[1:] - tt[:-1])).sum()
        if system == "A":
            if k + 1 < nc:
                o_end, t_end = out_f[i1], t_f[i1]
            elif closing is not None:
                o_end, t_end = closing[0], torch.as_tensor(closing[1], dtype=t.dtype)
            else:
                break
            W = W + 0.5 * (y[-1] + f(o_end, r)) * (t_end - tt[-1])
        else:
            t_end = t_f[i1] if k + 1 < nc else tt[-1] + (tt[-1] - tt[-2])
            W = W + y[-1] * (t_end - tt[-1])
        vals.append(W)
    return torch.stack(vals) if vals else torch.zeros(0, dtype=t.dtype)


def relative_error(pred, ref):
    return (pred - ref).abs() / ref.abs()


def apply_failure(e, eps, finite):
    """(e, eps, finite) per cycle -> (e capped, eps capped, failed cycle 1-based or None)."""
    bad = (~finite) | ~torch.isfinite(e) | ~torch.isfinite(eps) | (e > CAP) | (eps > CAP)
    e, eps = e.clone(), eps.clone()
    if not bool(bad.any()):
        return e, eps, None
    k = int(torch.nonzero(bad)[0])
    e[k:] = CAP
    eps[k:] = CAP
    return e, eps, k + 1
