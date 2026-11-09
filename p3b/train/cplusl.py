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


def _loss(data: _Data, n_windows, n_steps, K, rng):
    total = 0.0
    for _ in range(n_windows):
        j = int(rng.integers(len(data.Y)))
        Y = data.Y[j].reshape(-1, data.Y[j].shape[-1])
        s = int(rng.integers(Y.shape[0] - n_steps))
        y = Y[s]
        for k in range(n_steps):
            i = s + k
            y = implicit_euler_step(data.rhs[j], y, data.raw[j][i, :3], float(data.t[j][i + 1] - data.t[j][i]),
                                    substeps=K)
            total = total + (((y - Y[i + 1]) / data.sd) ** 2).mean()
    return total / (n_windows * n_steps)


def cplusl_loss(mult, ds, material, n_coarse, n_windows, n_steps, K, rng):
    with torch.no_grad():
        return _loss(_Data(ds, material, n_coarse, mult), n_windows, n_steps, K, rng)


def train_cplusl(mult, ds, material: str, n_coarse: int, lr: float, iterations: int, seed: int,
                 n_windows: int = 8, n_steps: int = 20, K: int = 1, lr_min: float = 1e-5) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = _Data(ds, material, n_coarse, mult)
    params = [p for p in mult.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    hist = {"loss": [], "skipped": 0}
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
            continue
        opt.zero_grad()
        loss.backward()
        if not all(bool(torch.isfinite(p.grad).all()) for p in params if p.grad is not None):
            hist["skipped"] += 1
            hist["loss"].append(float(loss.detach()))
            continue
        opt.step()
        hist["loss"].append(float(loss.detach()))
    return hist


@torch.no_grad()
def rollout_cplusl_to_file(mult, point: dict, material: str, n_coarse: int, n_target: int, n_snap: int,
                           K: int, path) -> None:
    """C+L from the pre-wetted bed over point["n_cycles"] cycles, stored at the target cells."""
    col, schedule = setup_a(point, material, n_coarse)
    rhs = TorchColumnRHS(col, mult)
    c_in = schedule[0][2][0]
    y = torch.tensor(prewet_state(col, c_in, float(point["T_ads"])), dtype=F64)
    model, _ = setup_a(point, material, n_target)
    cz = Coarsener("A", n_target, n_target, model=model, c_in=c_in)
    half = n_snap // 2
    t_abs = 0.0
    for k in range(int(point["n_cycles"])):
        ys, ts = [], []
        for _, dur, inputs in schedule:
            h = dur / half
            inp = torch.tensor(inputs, dtype=F64)
            for i in range(half):
                ys.append(y.numpy().copy())
                ts.append(t_abs + i * h)
                y = implicit_euler_step(rhs, y, inp, h, substeps=K)
            t_abs += dur
        Y = np.stack(ys)
        aug = np.concatenate([Y, np.zeros((Y.shape[0], n_coarse))], axis=1)    # no wall-heat states
        cz.add_cycle(k, np.array(ts), map_to_target(aug, 3, n_coarse, n_target // n_coarse), np.zeros((len(ts), 7)))
    cz.to_npz(path, meta=dict(arm="C+L", point=point, material=material, schedule=schedule, n_snap=n_snap,
                              N_solver=n_coarse, substeps=K))
