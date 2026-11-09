"""Long rollouts of the learned arms (PREREG Sections 2 and 5; Plan 3, Task 4).

A rollout starts from the encoded first stored snapshot of the trajectory and integrates every
snapshot interval with that interval's inputs (piecewise constant), h_steps RK4 steps each
(Arm.step), decoding the state at every snapshot; the last interval of the last cycle is
integrated too, so the release of the last cycle can be closed (X_end). Non-finite states end the
rollout: the remaining snapshots are NaN and the failed cycle (1-based) is recorded; e and epsilon
are then capped by p3b.eval.metrics.apply_failure. The projected arms' defect flags are summed.
Arms C and MUSCL-C are not rolled out here: their stored files (p3b.solvers.coarse.run_c) are read
like the reference.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


@dataclass
class RolloutResult:
    X: torch.Tensor                # (cycles, snaps, n_state), unstandardised; NaN after a failure
    X_end: torch.Tensor            # the state at the end of the last cycle
    finite: torch.Tensor           # (cycles,) bool
    failed_cycle: int | None
    cpu_s: float
    flags: dict = field(default_factory=dict)
    residuals: dict = field(default_factory=dict)  # per interval: N, E (balances) and S (entropy)


@torch.no_grad()
def rollout(arm, ds, j: int, n_cycles: int, h_steps: int, track: bool = False) -> RolloutResult:
    std = arm.std
    ns, n = ds.X.shape[2], ds.X.shape[-1]
    T = n_cycles * ns
    X0 = ds.X[j, 0, 0]
    F = std.encode_features(ds.features[j, :n_cycles].reshape(T, -1))
    R = ds.raw_inputs[j, :n_cycles].reshape(T, -1)
    t = ds.t[j, :n_cycles].reshape(T)
    dt = torch.empty(T, dtype=t.dtype)
    dt[:-1] = t[1:] - t[:-1]
    dt[-1] = t[-1] - t[-2]                                   # uniform spacing within a half-cycle
    decode = (lambda z: std.decode(z)) if arm.kind == "FNO" else (lambda z: std.decode(arm.ae.decode(z)))
    z = std.encode(X0)[None] if arm.kind == "FNO" else arm.ae.encode(std.encode(X0)[None])
    out = torch.full((T + 1, n), float("nan"), dtype=X0.dtype)
    out[0] = decode(z)[0]
    flags: dict = {}
    residuals: dict = {}
    failed_at = None
    t0 = time.process_time()
    for i in range(T):
        z_prev = z
        with torch.enable_grad() if arm.kind in ("S", "S-M", "S-Pi", "S-L", "U+F") else torch.no_grad():
            z, fl = arm.step(z, F[i:i + 1], R[i:i + 1], dt[i:i + 1, None], h_steps, track=track)
        z = z.detach()
        if fl is not None:
            for k, v in fl.items():
                if v.dtype == torch.bool:
                    flags[k] = flags.get(k, 0) + int(v.sum())
            if track and "Q_E" in fl:
                v0 = arm.lf.evaluate(z_prev, R[i:i + 1], grads=False)
                v1 = arm.lf.evaluate(z, R[i:i + 1], grads=False)
                for q in ("N", "E", "S"):
                    if "Q_" + q in fl:
                        residuals.setdefault(q, []).append(float(v1[q] - v0[q] - fl["Q_" + q].detach()))
        x = decode(z)[0]
        if not bool(torch.isfinite(x).all()):
            failed_at = i + 1                                   # snapshot index that failed
            break
        out[i + 1] = x
    cpu = time.process_time() - t0
    X = out[:T].reshape(n_cycles, ns, n)
    finite = torch.isfinite(X).reshape(n_cycles, -1).all(-1)
    if failed_at is not None and failed_at == T:                # only the closing state failed
        finite[-1] = False
    failed_cycle = None if bool(finite.all()) else int(torch.nonzero(~finite)[0]) + 1
    return RolloutResult(X, out[T], finite, failed_cycle, cpu, flags,
                         {k: torch.tensor(v, dtype=torch.float64) for k, v in residuals.items()})
