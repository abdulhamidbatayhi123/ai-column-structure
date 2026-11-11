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
from p3b.solvers.c_plus_l import TorchColumnRHS, implicit_euler_step
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

    def steps(self, picks, n_steps, K):
        """Integrate windows (trajectory j, first snapshot s) together, one batched Newton solve per
        snapshot interval; yields (step k, states, targets), each (windows, 3 n_coarse)."""
        rhs = TorchColumnRHS.stack([self.rhs[j] for j, _ in picks])
        y = torch.stack([self.flat[j][s] for j, s in picks])
        cache = {}                                     # Newton's Jacobian, carried between intervals
        for k in range(n_steps):
            inp = torch.stack([torch.as_tensor(self.raw[j][s + k, :3], dtype=F64) for j, s in picks])
            h = torch.tensor([float(self.t[j][s + k + 1] - self.t[j][s + k]) for j, s in picks], dtype=F64)
            y = implicit_euler_step(rhs, y, inp, h, substeps=K, jac_cache=cache)
            yield k, y, torch.stack([self.flat[j][s + k + 1] for j, s in picks])


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
    except RuntimeError:                                 # Newton failure: not finished
        return float("nan")
    return float((tot / (ns - 1)).mean())


def train_cplusl(mult, ds, material: str, n_coarse: int, lr: float, iterations: int, seed: int,
                 n_windows: int = 8, n_steps: int = 20, K: int = 1, lr_min: float = 1e-5, every: int = 0,
                 callback=None) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = _Data(ds, material, n_coarse, mult)
    params = [p for p in mult.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    hist = {"loss": [], "skipped": 0, "skipped_grad": 0}
    for it in range(iterations):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(it, max(iterations - 1, 1), lr, lr_min)
        try:
            loss = _loss(data, n_windows, n_steps, K, rng)
        except RuntimeError:                         # Newton failure: skipped and counted, never applied
            loss = None
        if loss is None or not math.isfinite(float(loss)):
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
                            K: int, paths: list) -> None:
    """rollout_cplusl_to_file for several trajectories of one material and cycle count, integrated
    together (one batched Newton solve per snapshot interval); each is stored to its own path."""
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
    for k in range(n_cycles.pop()):
        ys, ts = [], []
        for ph in range(len(setups[0][1])):
            durs = np.array([sch[ph][1] for _, sch in setups], dtype=float)
            h = durs / half
            inp = torch.tensor([sch[ph][2] for _, sch in setups], dtype=F64)
            for i in range(half):
                ys.append(y.numpy().copy())
                ts.append(t_abs + i * h)
                y = implicit_euler_step(rhs, y, inp, torch.tensor(h, dtype=F64), substeps=K, jac_cache=cache)
            t_abs = t_abs + durs
        Y, T = np.stack(ys, axis=1), np.stack(ts, axis=1)                     # (B, snaps, ...)
        for b, cz in enumerate(czs):
            aug = np.concatenate([Y[b], np.zeros((Y.shape[1], n_coarse))], axis=1)   # no wall-heat states
            cz.add_cycle(k, T[b], map_to_target(aug, 3, n_coarse, n_target // n_coarse), np.zeros((Y.shape[1], 7)))
    for cz, (_, sch), pt, path in zip(czs, setups, points, paths):
        cz.to_npz(path, meta=dict(arm="C+L", point=pt, material=material, schedule=sch, n_snap=n_snap,
                                  N_solver=n_coarse, substeps=K))


def rollout_cplusl_each(mult, points: list, material: str, n_coarse: int, n_target: int, n_snap: int,
                        K: int, paths: list) -> list:
    """Evaluation rollouts: the paths written, None where a trajectory's Newton solve failed. The
    trajectories are integrated together; if the batch fails, each is integrated alone, so a failure
    is that trajectory's own and the others are stored as before batching."""
    from p3b.train import cplusl as _self                # the module attribute (patchable in tests)
    out = [None] * len(points)
    groups = {}
    for j, pt in enumerate(points):                      # one batch per cycle count
        groups.setdefault(int(pt["n_cycles"]), []).append(j)
    for idx in groups.values():
        try:
            _self.rollout_cplusl_to_files(mult, [points[j] for j in idx], material, n_coarse, n_target, n_snap, K,
                                          paths=[paths[j] for j in idx])
            for j in idx:
                out[j] = paths[j]
            continue
        except RuntimeError:
            pass
        for j in idx:
            try:
                _self.rollout_cplusl_to_files(mult, [points[j]], material, n_coarse, n_target, n_snap, K,
                                              paths=[paths[j]])
                out[j] = paths[j]
            except RuntimeError:                         # Newton failure: a failure at cycle 1
                pass
    return out
