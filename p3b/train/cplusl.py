"""Training and rolling out arm C+L (PREREG Section 3; plan 3 review I6; Plan 4 Task 4b).

C+L lives in the solver's state space: n, a, u on 32 cells. It is trained on the same coarse data
as every arm, in its own variables: the stored conserved block averages (128 cells) averaged onto
its 32 cells, each variable standardised over the training set, MSE along windows of snapshot
intervals, each interval integrated by implicit Euler with K substeps (K from the pilot). Every
trajectory has its own column physics (v, D_L, k_z, k_LDF); the multipliers are shared. For
evaluation it starts, like C, from the pre-wetted bed and integrates every cycle; its states are
mapped to the target cells by the conservative minmod reconstruction and stored by the reference
Coarsener, so p3b.eval.evaluate.evaluate_files scores it like C.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from p3b.latent.autoencoder import cosine_lr
from p3b.simulate_a import prewet_state, setup_a
from p3b.solvers.c_plus_l import SOLVER_FAILURES, TorchColumnRHS, implicit_euler_step
from p3b.solvers.coarse import map_to_target
from p3b.store import Coarsener

F64 = torch.float64


def coarse_states(paths, n_coarse: int) -> list:
    """Per trajectory (cycles, snaps, 3 n_coarse): [n | a | u] block averages onto n_coarse cells."""
    out = []
    for p in paths:
        cb = np.load(p)["conserved_blocks"]                       # (cycles, snaps, NC, 3)
        nc, ns, NC, _ = cb.shape
        r = NC // n_coarse
        blk = cb.reshape(nc, ns, n_coarse, r, 3).mean(3)          # (cycles, snaps, n_coarse, 3)
        out.append(torch.tensor(np.concatenate([blk[..., v] for v in range(3)], axis=-1), dtype=F64))
    return out


class _Data:
    def __init__(self, ds, material, n_coarse, mult):
        self.Y = coarse_states(ds.paths, n_coarse)
        allY = torch.cat([y.reshape(-1, y.shape[-1]) for y in self.Y])
        blocks = allY.reshape(allY.shape[0], 3, n_coarse)
        self.mean = blocks.mean(dim=(0, 2)).repeat_interleave(n_coarse)
        sd = blocks.std(dim=(0, 2))
        self.sd = torch.where(sd > 0, sd, torch.ones_like(sd)).repeat_interleave(n_coarse)
        self.rhs = [TorchColumnRHS(setup_a(m["point"], material, n_coarse)[0], mult) for m in ds.meta]
        self.raw = [ds.raw_inputs[j].reshape(-1, ds.raw_inputs.shape[-1]) for j in range(len(ds.meta))]
        self.t = [ds.t[j].reshape(-1) for j in range(len(ds.meta))]
        self.flat = [y.reshape(-1, y.shape[-1]) for y in self.Y]       # (snapshots, 3 n_coarse) per trajectory
        self.retries = 0                                # substeps re-solved by full Newton (all calls)

    def steps(self, picks, n_steps, K):
        """Integrate windows (trajectory j, first snapshot s) together, one batched Newton solve per
        snapshot interval; yields (step k, states, targets), each (windows, 3 n_coarse)."""
        rhs = TorchColumnRHS.stack([self.rhs[j] for j, _ in picks])
        y = torch.stack([self.flat[j][s] for j, s in picks])
        cache = {}                                     # Newton's Jacobian, carried between intervals
        try:
            for k in range(n_steps):
                inp = torch.stack([torch.as_tensor(self.raw[j][s + k, :3], dtype=F64) for j, s in picks])
                h = torch.tensor([float(self.t[j][s + k + 1] - self.t[j][s + k]) for j, s in picks], dtype=F64)
                y = implicit_euler_step(rhs, y, inp, h, substeps=K, jac_cache=cache)
                yield k, y, torch.stack([self.flat[j][s + k + 1] for j, s in picks])
        finally:                                       # counted also when a solve fails
            self.retries += cache.get("full_newton_retries", 0)


def _loss(data: _Data, n_windows, n_steps, K, rng):
    """MSE along n_windows random windows of n_steps snapshot intervals (the windows drawn in the
    same order as before batching: trajectory, then first snapshot), integrated together."""
    picks = []
    for _ in range(n_windows):
        j = int(rng.integers(len(data.Y)))
        picks.append((j, int(rng.integers(data.flat[j].shape[0] - n_steps))))
    total = 0.0
    for _, y, target in data.steps(picks, n_steps, K):
        total = total + (((y - target) / data.sd) ** 2).mean(-1).sum()
    return total / (n_windows * n_steps)


def cplusl_loss(mult, ds, material, n_coarse, n_windows, n_steps, K, rng):
    with torch.no_grad():
        return _loss(_Data(ds, material, n_coarse, mult), n_windows, n_steps, K, rng)


@torch.no_grad()
def cplusl_val_loss(mult, val, material: str, n_coarse: int, K: int) -> float:
    """C+L's deterministic validation loss (the counterpart of the latent arms' one-cycle rollout
    loss): from each validation trajectory's first stored coarse state, one cycle of snapshot
    intervals by implicit Euler, standardised MSE in its own variables; NaN if a Newton solve fails."""
    data = _Data(val, material, n_coarse, mult)
    ns = val.X.shape[2]
    tot = torch.zeros(len(data.Y), dtype=F64)
    try:                                                 # every trajectory together, from its first state
        for _, y, target in data.steps([(j, 0) for j in range(len(data.Y))], ns - 1, K):
            tot = tot + (((y - target) / data.sd) ** 2).mean(-1)
    except SOLVER_FAILURES:                              # a failed solve: not finished (other errors propagate)
        return float("nan")
    return float((tot / (ns - 1)).mean())


def train_cplusl(mult, ds, material: str, n_coarse: int, lr: float, iterations: int, seed: int,
                 n_windows: int = 8, n_steps: int = 20, K: int = 1, lr_min: float = 1e-5, every: int = 0,
                 callback=None, checkpoint: str | None = None, deadline: float | None = None,
                 stop_after: int | None = None, save_every: int = 200) -> dict:
    """Sessions (as p3b.train.trainer.train): with `checkpoint`, a run resumes from the file if it exists
    (weights, optimiser, both random states, history), saves it every `save_every` iterations, and past
    `deadline` (time.monotonic(); at least one iteration per session) or after `stop_after` iterations
    saves it and returns with hist["stopped_at"]. A resumed run equals the run done in one go."""
    import os
    import time
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = _Data(ds, material, n_coarse, mult)
    params = [p for p in mult.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    # newton_failures: [iteration, reason] of every iteration skipped for a failed solve
    hist = {"loss": [], "skipped": 0, "skipped_grad": 0, "newton_failures": [], "full_newton_retries": 0}
    it0 = 0
    if checkpoint and os.path.exists(checkpoint):
        ck = torch.load(checkpoint, weights_only=False)
        mult.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"])
        hist.update(ck["hist"])
        it0 = ck["it"]
        hist["resumed_from"] = it0

    def save(it):
        if checkpoint:
            tmp = checkpoint + ".partial"
            torch.save({"model": mult.state_dict(), "opt": opt.state_dict(), "rng": rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state(), "it": it,
                        "hist": {k: v for k, v in hist.items() if k not in ("resumed_from", "stopped_at")}}, tmp)
            os.replace(tmp, checkpoint)

    done_here = 0
    for it in range(it0, iterations):
        if (stop_after is not None and done_here >= stop_after) or (
                deadline is not None and done_here >= 1 and time.monotonic() >= deadline):
            save(it)
            hist["stopped_at"] = it
            return hist
        done_here += 1
        if checkpoint and it > it0 and it % save_every == 0:
            save(it)
        for g in opt.param_groups:
            g["lr"] = cosine_lr(it, max(iterations - 1, 1), lr, lr_min)
        r0 = data.retries
        try:
            loss = _loss(data, n_windows, n_steps, K, rng)
        except SOLVER_FAILURES as exc:               # a failed solve: skipped and counted, never applied
            loss = None
            hist["newton_failures"].append([it, f"{type(exc).__name__}: {exc}"])
        hist["full_newton_retries"] += data.retries - r0
        if loss is None or not math.isfinite(float(loss.detach())):
            hist["skipped"] += 1
            hist["loss"].append(float("nan"))
            _checkpoint(it, every, callback, mult, hist)
            continue
        opt.zero_grad()
        loss.backward()
        if not all(bool(torch.isfinite(p.grad).all()) for p in params if p.grad is not None):
            hist["skipped_grad"] += 1
            hist["loss"].append(float(loss.detach()))
            _checkpoint(it, every, callback, mult, hist)
            continue
        opt.step()
        hist["loss"].append(float(loss.detach()))
        _checkpoint(it, every, callback, mult, hist)
    return hist


def _checkpoint(it: int, every: int, callback, mult, hist) -> None:
    """callback(iterations done, mult, hist so far) after every `every` iterations (the pilot's curves;
    hist carries the skipped counts, review I5)."""
    if callback is not None and every and (it + 1) % every == 0:
        callback(it + 1, mult, hist)


@torch.no_grad()
def rollout_cplusl_to_file(mult, point: dict, material: str, n_coarse: int, n_target: int, n_snap: int,
                           K: int, path) -> None:
    """C+L from the pre-wetted bed over point["n_cycles"] cycles, stored at the target cells."""
    rollout_cplusl_to_files(mult, [point], material, n_coarse, n_target, n_snap, K, [path])


@torch.no_grad()
def rollout_cplusl_to_files(mult, points: list, material: str, n_coarse: int, n_target: int, n_snap: int,
                            K: int, paths: list, keep_partial: bool = False) -> dict:
    """rollout_cplusl_to_file for several trajectories of one material and cycle count, integrated
    together (one batched Newton solve per snapshot interval); each is stored to its own path.

    A failed solve (NewtonFailure, LinAlgError) in cycle k propagates, unless keep_partial is set and
    the batch is one trajectory: then its completed cycles 1..k-1 are stored (no file when k = 1), with
    failed_cycle and failure_reason in the meta, so it is scored from cycle k on (PREREG 5). Returns
    {"failed_cycle": [1-based or None per trajectory], "reason": [...], "full_newton_retries": n}."""
    n_cycles = {int(pt["n_cycles"]) for pt in points}
    if len(n_cycles) != 1:
        raise ValueError("a batch of rollouts shares one cycle count")
    setups = [setup_a(pt, material, n_coarse) for pt in points]
    if len({len(sch) for _, sch in setups}) != 1:
        raise ValueError("a batch of rollouts shares one schedule structure")
    rhs = TorchColumnRHS.stack([TorchColumnRHS(col, mult) for col, _ in setups])
    c_ins = [sch[0][2][0] for _, sch in setups]
    y = torch.stack([torch.tensor(prewet_state(col, c_in, float(pt["T_ads"])), dtype=F64)
                     for (col, _), c_in, pt in zip(setups, c_ins, points)])
    czs = [Coarsener("A", n_target, n_target, model=setup_a(pt, material, n_target)[0], c_in=c_in)
           for pt, c_in in zip(points, c_ins)]
    half = n_snap // 2
    B = len(points)
    t_abs = np.zeros(B)
    cache = {}                                         # Newton's Jacobian, carried between intervals
    failed = None                                      # (1-based cycle, reason) of a kept partial run
    for k in range(n_cycles.pop()):
        ys, ts = [], []
        try:
            for ph in range(len(setups[0][1])):
                durs = np.array([sch[ph][1] for _, sch in setups], dtype=float)
                h = durs / half
                inp = torch.tensor([sch[ph][2] for _, sch in setups], dtype=F64)
                for i in range(half):
                    ys.append(y.numpy().copy())
                    ts.append(t_abs + i * h)
                    y = implicit_euler_step(rhs, y, inp, torch.tensor(h, dtype=F64), substeps=K, jac_cache=cache)
                t_abs = t_abs + durs
        except SOLVER_FAILURES as exc:
            if not (keep_partial and B == 1):          # a batch: the caller reruns each trajectory alone
                raise
            failed = (k + 1, f"{type(exc).__name__}: {exc}")
            break
        Y, T = np.stack(ys, axis=1), np.stack(ts, axis=1)                     # (B, snaps, ...)
        for b, cz in enumerate(czs):
            aug = np.concatenate([Y[b], np.zeros((Y.shape[1], n_coarse))], axis=1)   # no wall-heat states
            cz.add_cycle(k, T[b], map_to_target(aug, 3, n_coarse, n_target // n_coarse), np.zeros((Y.shape[1], 7)))
    for cz, (_, sch), pt, path in zip(czs, setups, points, paths):
        meta = dict(arm="C+L", point=pt, material=material, schedule=sch, n_snap=n_snap, N_solver=n_coarse,
                    substeps=K)
        if failed is None:
            cz.to_npz(path, meta=meta)
        elif cz.cycles:                                # the cycles before the failing one
            cz.to_npz(path, meta=dict(meta, failed_cycle=failed[0], failure_reason=failed[1]))
    return {"failed_cycle": [None if failed is None else failed[0]] * B,
            "reason": [None if failed is None else failed[1]] * B,
            "full_newton_retries": int(cache.get("full_newton_retries", 0))}


def rollout_cplusl_each(mult, points: list, material: str, n_coarse: int, n_target: int, n_snap: int,
                        K: int, paths: list, report: dict | None = None) -> list:
    """Evaluation rollouts: per trajectory the path written, None where its solve failed in cycle 1.
    The trajectories are integrated together; if the batch fails, each is integrated alone, so a
    failure is that trajectory's own and the others are stored as before batching. A trajectory that
    fails in cycle k > 1 keeps cycles 1..k-1 in its file and is scored from cycle k on (PREREG 5).

    `report` (a dict) receives batch_fallback [{trajectories, reason}], failures {j: {failed_cycle,
    reason}} and full_newton_retries. Only failed solves are caught; any other error propagates."""
    from p3b.train import cplusl as _self                # the module attribute (patchable in tests)
    rep = report if report is not None else {}
    rep.setdefault("batch_fallback", [])
    rep.setdefault("failures", {})
    rep.setdefault("full_newton_retries", 0)
    out = [None] * len(points)
    groups = {}
    for j, pt in enumerate(points):                      # one batch per cycle count
        groups.setdefault(int(pt["n_cycles"]), []).append(j)
    for idx in groups.values():
        try:
            runs = [(idx, _self.rollout_cplusl_to_files(mult, [points[j] for j in idx], material, n_coarse,
                                                        n_target, n_snap, K, paths=[paths[j] for j in idx],
                                                        keep_partial=True))]
        except SOLVER_FAILURES as exc:
            rep["batch_fallback"].append({"trajectories": list(idx), "reason": f"{type(exc).__name__}: {exc}"})
            runs = []
            for j in idx:
                try:
                    runs.append(([j], _self.rollout_cplusl_to_files(mult, [points[j]], material, n_coarse, n_target,
                                                                    n_snap, K, paths=[paths[j]], keep_partial=True)))
                except SOLVER_FAILURES as exc1:          # failed before a cycle was kept: cycle 1
                    runs.append(([j], {"failed_cycle": [1], "reason": [f"{type(exc1).__name__}: {exc1}"]}))
        for js, info in runs:
            info = info or {}
            rep["full_newton_retries"] += int(info.get("full_newton_retries", 0))
            for j, fc, why in zip(js, info.get("failed_cycle", [None] * len(js)), info.get("reason", [None] * len(js))):
                if fc is None or fc > 1:
                    out[j] = paths[j]
                if fc is not None:
                    rep["failures"][j] = {"failed_cycle": int(fc), "reason": why}
    return out
