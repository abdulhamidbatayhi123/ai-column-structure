"""The decoder-only floor (PREREG Section 6, revision R1): epsilon_W and e with no learned dynamics.

R1's balance term targets the decoded totals E^ and N^, which the projected arms use and U does not, so it
can lower the projected arms' error floor without lowering U's. These floors show how much of each arm's
error the autoencoder alone sets, per test trajectory and cycle:

- `u_floor`: every stored reference state encoded and decoded (what U, U+F and P could reach with exact
  latent dynamics; their released amount comes from the decoded outlet);
- `projected_floor`: the encoded reference path, interval by interval, through the projected arms'
  balance projection. Per snapshot interval i the latent field is the constant
  c_i = (enc(x_{i+1}) - enc(x_i)) / dt_i, stepped by the production code (`Arm.step` as U+proj, inside
  `rollout_batch`): h_steps RK4 steps, each followed by the Gauss-Newton projection onto the discrete
  water and energy balances with the port integrals on the same stages, the arm's own LatentFunctionals.
  RK4 is exact for a constant field, so with the projection off (`project=False`) this reproduces the
  round trip up to rounding (tests/test_decoder_floor.py).

Both are scored by `p3b.eval.evaluate.evaluate`, the arms' own metric code, so a floor row has the same
schema and the same e, epsilon, capping and violation rules as an arm's row. Costs no training.
"""
from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from p3b.eval import rollout as ro
from p3b.eval.evaluate import evaluate, write_results
from p3b.eval.metrics import release
from p3b.latent.data import Standardiser
from p3b.train.trainer import Arm


class _IndexStd(Standardiser):
    """The arm's standardiser, with the features passed through: the floor's features are indices."""

    def __init__(self, std):
        super().__init__(std.mean, std.sd, std.f_mean, std.f_sd)

    def encode_features(self, F):
        return F


class _ConstantField(nn.Module):
    """The latent field of interval i of trajectory j: inc[j, i], looked up from the (j, i) feature."""

    def __init__(self, inc):
        super().__init__()
        self.inc = inc                                   # (traj, T, d)

    def forward(self, z, feat):
        j, i = feat[:, 0].round().long(), feat[:, 1].round().long()
        return self.inc[j, i].to(z.dtype)


def _dt(ds, j, n_cycles):
    t = ds.t[j, :n_cycles].reshape(-1)
    dt = torch.empty_like(t)
    dt[:-1] = t[1:] - t[:-1]
    dt[-1] = t[-1] - t[-2]                               # as in rollout: the closing interval
    return dt


@torch.no_grad()
def _increments(ae, std, ds, n_cycles):
    """(traj, T, d): the encoded reference's change per unit time over each snapshot interval; the last
    interval runs to the first snapshot of the next stored cycle, or is zero if there is none."""
    n = ds.X.shape[-1]
    out = []
    for j in range(ds.X.shape[0]):
        Z = ae.encode(std.encode(ds.X[j, :n_cycles].reshape(-1, n)))
        if ds.X.shape[1] > n_cycles:
            z_end = ae.encode(std.encode(ds.X[j, n_cycles, 0][None]))
        else:
            z_end = Z[-1:]
        Z = torch.cat([Z, z_end])
        out.append((Z[1:] - Z[:-1]) / _dt(ds, j, n_cycles)[:, None])
    return torch.stack(out)


def _index_view(ds):
    """The data set with its features replaced by (trajectory, flat interval) indices."""
    n_traj, n_cyc, ns = ds.X.shape[:3]
    j = torch.arange(n_traj, dtype=torch.float64)[:, None, None].expand(n_traj, n_cyc, ns)
    i = torch.arange(n_cyc * ns, dtype=torch.float64).reshape(1, n_cyc, ns).expand(n_traj, n_cyc, ns)
    return replace(ds, features=torch.stack([j, i], -1))


def _arm(kind, ae, std, lf, inc):
    return Arm(kind, _ConstantField(inc), ae, _IndexStd(std), lf)


def projected_rollouts(ae, std, lf, ds, n_cycles: int, h_steps: int, batch: int = ro.BATCH,
                       project: bool = True, track: bool = True) -> list:
    """The projected floor's RolloutResults (batched like the arms' evaluation)."""
    arm = _arm("U+proj" if project else "U", ae, std, lf, _increments(ae, std, ds, n_cycles))
    return ro.rollouts(arm, _index_view(ds), n_cycles, h_steps, track=track, batch=batch)


def projected_floor(ae, std, lf, ds, phys, n_cycles: int, h_steps: int, floors: dict, tau_S: float,
                    seed: int = 0, project: bool = True) -> list:
    arm = _arm("U+proj" if project else "U", ae, std, lf, _increments(ae, std, ds, n_cycles))
    return evaluate(arm, _index_view(ds), phys, n_cycles, h_steps, floors, tau_S, seed)


@torch.no_grad()
def _round_trip(ae, std, ds, j, n_cycles):
    n = ds.X.shape[-1]
    X = ds.X[j, :n_cycles]
    return std.decode(ae.decode(ae.encode(std.encode(X.reshape(-1, n))))).reshape(X.shape)


def _release_of(X, ds, j, n_cycles, phys):
    """W_k (A) or Q_k (B) of a predicted state sequence, by the evaluation's rule."""
    n_out = 2 if ds.system == "A" else 1
    rhoc = None if ds.system == "A" else phys.rhoc
    return [float(x) for x in release(X[..., -n_out:], ds.raw_inputs[j, :n_cycles], ds.t[j, :n_cycles],
                                      ds.system, rhoc)]


def u_floor(ae, std, lf, ds, phys, n_cycles: int, floors: dict, tau_S: float, seed: int = 0) -> list:
    """Rows of the round-trip floor (no dynamics, no violation counts: those need a stepped path)."""
    results = []
    for j in range(ds.X.shape[0]):
        X = _round_trip(ae, std, ds, j, n_cycles)
        results.append(ro.RolloutResult(X, X[-1, -1], torch.ones(n_cycles, dtype=torch.bool), None, 0.0))
    arm = _arm("U", ae, std, lf, None)
    return evaluate(arm, ds, phys, n_cycles, 1, floors, tau_S, seed, results=results)


def decoder_floor(ae, std, lf, ds, phys, n_cycles: int, h_steps: int, floors: dict, tau_S: float,
                  seed: int = 0) -> dict:
    """Both floors for one autoencoder (with its LatentFunctionals) on one set."""
    return {"U": u_floor(ae, std, lf, ds, phys, n_cycles, floors, tau_S, seed),
            "projected": projected_floor(ae, std, lf, ds, phys, n_cycles, h_steps, floors, tau_S, seed)}


def write_floor(path, both: dict, meta: dict) -> None:
    """One results file per autoencoder and set, beside the arms' (rows tagged by floor)."""
    rows = [dict(r, floor=k) for k in ("U", "projected") for r in both[k]]
    write_results(path, rows, dict(meta, kind="decoder_floor"))
