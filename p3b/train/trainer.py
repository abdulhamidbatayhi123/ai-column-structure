"""One trainer for the learned arms (PREREG Sections 3 and 4; Plan 3, Task 3).

Every arm is trained on multi-step rollouts: batches of 8 windows x 20 snapshot intervals; the
loss is the MSE of the decoded standardised channels along each window, outlet included, equal
weights; Adam with the arm's learning rate and cosine decay to 1e-5 over one fixed iteration
count; the final checkpoint is used, with no early stopping. Latent arms start from the encoded
first state of the window and integrate with RK4 at h = dt / h_steps (per sample, since windows
differ in snapshot spacing); the projected arms (S, S-M, S-Pi, S-L) use `structured_step`, whose
defect flags are counted. The FNO steps the standardised state directly. P adds the penalty with
one weight fixed at iteration 0 so that it equals 1 % of the data loss there; its water and energy
terms are divided by the violation-metric normalisations and its entropy term by its own initial
RMS, so the three terms are dimensionless. U+F receives its physical terms standardised on
training latents (TermsStandardiser). The autoencoder is frozen throughout.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from p3b.eval.violations import normalisers, step_residuals
from p3b.latent.arms_structured import projected_rk4_step, structured_step
from p3b.latent.arms_unstructured import count_params, penalty_loss, penalty_weight
from p3b.latent.autoencoder import cosine_lr
from p3b.latent.data import windows
from p3b.latent.integrate import rk4_step, stage_quadrature

PROJECTED = ("S", "S-M", "S-Pi", "S-L")
KINDS = ("U", "U+F", "P", "S", "S-M", "S-Pi", "S-L", "FNO", "U+proj", "U+F+proj")
FLAG_KEYS = ("rank_deficient", "kappa_binds", "projection_rank_deficient")


def time_scale(ds) -> float:
    """The reference time of the learned vector fields: the median snapshot interval of the
    training set (seconds). Networks output rates per reference time; integration is in seconds."""
    return float(torch.median(ds.t[..., 1:] - ds.t[..., :-1]))


@dataclass
class TrainConfig:
    lr: float
    iterations: int
    seed: int
    h_steps: int = 1
    n_windows: int = 8
    n_steps: int = 20
    lr_min: float = 1e-5


class TermsStandardiser:
    """Scalars (E^, N^, S^, ports) to zero mean and unit SD, latent gradients to unit mean norm,
    fitted on the latents of training snapshots."""

    SCALARS = ("E", "N", "S", "P_E", "P_N", "Phi_S")
    GRADS = ("dE", "dN", "dS")

    def __init__(self, stats):
        self.stats = stats

    @classmethod
    @torch.no_grad()
    def fit(cls, lf, ae, std, ds, n: int = 256, seed: int = 0):
        rng = np.random.default_rng(seed)
        X = ds.X.reshape(-1, ds.X.shape[-1])
        R = ds.raw_inputs.reshape(X.shape[0], -1)
        idx = torch.as_tensor(rng.choice(X.shape[0], size=min(n, X.shape[0]), replace=False))
        z = ae.encode(std.encode(X[idx]))
        t = lf.evaluate(z, R[idx], grads=True, create_graph=False)
        stats = {}
        for k in cls.SCALARS:
            if k in t:
                sd = float(t[k].std())
                stats[k] = (float(t[k].mean()), sd if sd > 0 else 1.0)
        for k in cls.GRADS:
            if k in t:
                nrm = float(t[k].norm(dim=-1).mean())
                stats[k] = (0.0, nrm if nrm > 0 else 1.0)
        return cls(stats)

    def transform(self, t: dict) -> dict:
        return {k: (v - self.stats[k][0]) / self.stats[k][1] if k in self.stats else v for k, v in t.items()}


class Arm:
    """A learned arm in training: its dynamics module and how it steps."""

    def __init__(self, kind: str, dyn, ae, std, lf, terms: TermsStandardiser | None = None):
        if kind not in KINDS:
            raise ValueError(f"unknown arm kind {kind!r}; known: {KINDS}")
        self.kind, self.dyn, self.ae, self.std, self.lf, self.terms = kind, dyn, ae, std, lf, terms
        if kind.startswith("U+F") and terms is None:
            raise ValueError("U+F needs a TermsStandardiser")

    def step(self, z, feat, raw, dt, h_steps: int, track: bool = False):
        """Advance one snapshot interval; returns (z, info or None). With `track`, info also holds
        the interval's port integrals Q_E, Q_N (A), Q_S summed over its RK4 steps, each taken on the
        arm's own stages (the discrete balance of PREREG 5), for the violation counts."""
        if self.kind == "FNO":
            return self.dyn(z, feat, dt[:, 0]), None
        h = dt / h_steps
        info = {}
        for _ in range(h_steps):
            if self.kind in PROJECTED:
                z, fl = structured_step(self.dyn, z, feat, raw, h)
            else:
                if self.kind.startswith("U+F"):
                    f = lambda x: self.dyn(x, feat, self.terms.transform(self.lf.evaluate(x, raw)))  # noqa: E731
                else:
                    f = lambda x: self.dyn(x, feat)  # noqa: E731
                if self.kind.endswith("+proj"):          # U+proj, U+F+proj: S's projection, no retraining
                    z, fl = projected_rk4_step(f, z, raw, h, self.lf)
                    fl = {k_: v for k_, v in fl.items()
                          if k_ in ("Q_E", "Q_N", "Q_S", "rank_deficient")}
                    for k_, v in fl.items():
                        if k_.startswith("Q_"):
                            info[k_] = info[k_] + v if k_ in info else v
                        else:
                            info["projection_" + k_] = info.get("projection_" + k_, v) | v
                    continue
                z_new, stages = rk4_step(f, z, h)
                fl = {}
                if track:
                    vals = [self.lf.evaluate(s_, raw, grads=False) for s_ in stages]
                    for n, port in (("E", "P_E"), ("N", "P_N"), ("S", "Phi_S")):
                        if port in vals[0]:
                            fl["Q_" + n] = stage_quadrature([v[port] for v in vals], h)
                z = z_new
            for k, v in fl.items():
                if k.startswith("Q_"):
                    info[k] = info[k] + v if k in info else v
                elif v.dtype == torch.bool:
                    info[k] = info[k] | v if k in info else v
                elif k in ("residual_deficit", "projection_residual"):
                    info[k] = torch.maximum(info[k], v) if k in info else v
        return z, (info or None)


def _penalty_terms(arm, z_prev, feat, raw, h):
    """Water, energy and entropy residuals of one RK4 step of U (for arm P)."""
    z_new, stages = rk4_step(lambda x: arm.dyn(x, feat), z_prev, h)
    r = step_residuals(arm.lf, z_prev, z_new, stages, h, raw)
    return z_new, r.get("N"), r["E"], r["S"] / h[:, 0]


def train(arm: Arm, ds, cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    params = [p for p in arm.dyn.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr)
    std = arm.std
    norm = normalisers(ds) if arm.kind == "P" else None
    hist = {"loss": [], "lr": [], "params": count_params(arm.dyn), "skipped": 0, "skipped_grad": 0,
            "flags": {k: 0 for k in FLAG_KEYS + ("projection_converged",)} if arm.kind in PROJECTED else {}}
    t0 = time.process_time()
    for it in range(cfg.iterations):
        lr = cosine_lr(it, max(cfg.iterations - 1, 1), cfg.lr, cfg.lr_min)
        for g in opt.param_groups:
            g["lr"] = lr
        x0, X, F, R, dt = windows(ds, cfg.n_windows, cfg.n_steps, rng)
        Fs, target = std.encode_features(F), std.encode(X)
        try:
            loss = _window_loss(arm, cfg, x0, Fs, R, dt, target, hist, norm, std)
        except torch.linalg.LinAlgError:
            loss = None
        if loss is None or not bool(torch.isfinite(loss)):
            hist["skipped"] += 1                      # a diverged window never updates the weights
            hist["loss"].append(float("nan"))
            hist["lr"].append(lr)
            continue
        opt.zero_grad()
        loss.backward()
        if not all(bool(torch.isfinite(p.grad).all()) for p in params if p.grad is not None):
            hist["skipped_grad"] += 1                 # a finite loss with a non-finite gradient: not applied
            opt.zero_grad()
            hist["loss"].append(float(loss.detach()))
            hist["lr"].append(lr)
            continue
        opt.step()
        hist["loss"].append(float(loss.detach()))
        hist["lr"].append(lr)
    hist["wall_s"] = time.process_time() - t0
    return hist


def _window_loss(arm, cfg, x0, Fs, R, dt, target, hist, norm, std):
    """The multi-step rollout loss of one batch of windows (plus P's penalty)."""
    z = std.encode(x0) if arm.kind == "FNO" else arm.ae.encode(std.encode(x0))
    data, pens = 0.0, []
    for k in range(cfg.n_steps):
        if arm.kind == "P":
            h = dt[:, k:k + 1] / cfg.h_steps
            for _ in range(cfg.h_steps):
                z, rN, rE, sig = _penalty_terms(arm, z, Fs[:, k], R[:, k], h)
                pens.append((rN, rE, sig))
            flags = None
        else:
            z, flags = arm.step(z, Fs[:, k], R[:, k], dt[:, k:k + 1], cfg.h_steps)
        pred = z if arm.kind == "FNO" else arm.ae.decode(z)
        data = data + ((pred - target[:, k]) ** 2).mean()
        if flags is not None:
            for key in FLAG_KEYS:
                hist["flags"][key] += int(flags[key].sum())
            hist["flags"]["projection_converged"] += int(flags["projection_converged"].sum())
    loss = data / cfg.n_steps
    if arm.kind == "P":
        rN = torch.cat([p[0] for p in pens]) if pens[0][0] is not None else None
        rE, sig = torch.cat([p[1] for p in pens]), torch.cat([p[2] for p in pens])
        sN = float(norm["feed_water"].mean()) if rN is not None else None
        sE = float(norm["energy_scale"].mean())
        if "penalty_scale_S" not in hist:
            hist["penalty_scale_S"] = float(sig.detach().pow(2).mean().sqrt()) or 1.0
        pen = penalty_loss(rN, rE, sig, scales=(sN, sE, hist["penalty_scale_S"]))
        if "penalty_weight" not in hist:
            hist["data_loss0"], hist["penalty0"] = float(loss.detach()), float(pen.detach())
            hist["penalty_weight"] = penalty_weight(hist["data_loss0"], hist["penalty0"])
        loss = loss + hist["penalty_weight"] * pen
    return loss
