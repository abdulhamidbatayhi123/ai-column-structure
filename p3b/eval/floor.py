"""The decoder-only floor (PREREG Section 6, revision R1): epsilon_W and e with no learned dynamics.

R1's balance term targets the decoded totals E^ and N^, which the projected arms use and U does not, so it
can lower the projected arms' error floor without lowering U's. These floors show how much of each arm's
error the autoencoder alone sets, per test trajectory and cycle:

- `u_floor`: every stored reference state encoded and decoded (what U, U+F and P could reach with exact
  latent dynamics; their released amount comes from the decoded outlet);
- `projected_floor`: per snapshot interval i, independently (no state is carried from one interval to
  the next), start from enc(x_i) and take ONE projected step of the production code (`Arm.step` as
  U+proj: h_steps RK4 substeps, each followed by the Gauss-Newton projection onto the discrete water and
  energy balances, ports on the same stages, the arm's own LatentFunctionals) with the constant latent
  field c_i = (enc(x_{i+1}) - enc(x_i)) / dt_i. The decoded result is the scored state at snapshot i+1;
  snapshot 0 is the round trip. Without the projection (`project=False`) RK4 is exact for a constant
  field, so this reproduces the round trip up to rounding. The interval after the last stored snapshot
  is never stepped or scored, so the violation counts cover the T - 1 intervals between stored snapshots.

Both are scored by `p3b.eval.evaluate.evaluate`, the arms' own metric code, so a floor row has the same
schema and the same e, epsilon, capping and violation rules as an arm's row. Costs no training; the
projected floor costs n_traj x (T - 1) projected steps per autoencoder and set, the same count as one
batched projected rollout of the set.
"""
from __future__ import annotations

import time

import torch
from torch import nn

from p3b.eval import rollout as ro
from p3b.eval.evaluate import evaluate, write_results
from p3b.eval.metrics import release
from p3b.train.trainer import Arm

CHUNK = 4096                        # (trajectory, interval) pairs per projected batch


class _FeatureField(nn.Module):
    """The latent field is the feature row itself: the floor passes c_i as the interval's feature."""

    def forward(self, z, feat):
        return feat.to(z.dtype)


def _arm(kind, ae, std, lf):
    return Arm(kind, _FeatureField(), ae, std, lf)


def _dt(ds, j, n_cycles):
    t = ds.t[j, :n_cycles].reshape(-1)
    return t[1:] - t[:-1]                                # the T - 1 intervals between stored snapshots


@torch.no_grad()
def _encode(ae, std, X):
    return ae.encode(std.encode(X))


@torch.no_grad()
def _decode(ae, std, Z):
    return std.decode(ae.decode(Z))


def projected_results(ae, std, lf, ds, n_cycles: int, h_steps: int, project: bool = True,
                      chunk: int = CHUNK) -> list:
    """RolloutResults of the per-interval projected floor, one per trajectory of ds. All (trajectory,
    interval) pairs are stepped in batches of `chunk`; a batch whose projection raises LinAlgError is
    redone one pair at a time, and only the pair that raises fails (its snapshot is NaN, so evaluate
    caps from its cycle on, as for an arm). Residuals (N, E, S) per interval are taken from the step's
    own anchor: lf(z_new) - lf(enc(x_i)) - Q on the step's stages (NaN for a failed pair)."""
    arm = _arm("U+proj" if project else "U", ae, std, lf)
    n_traj, ns, n = ds.X.shape[0], ds.X.shape[2], ds.X.shape[-1]
    T = n_cycles * ns
    Z = torch.stack([_encode(ae, std, ds.X[j, :n_cycles].reshape(T, n)) for j in range(n_traj)])
    dt = torch.stack([_dt(ds, j, n_cycles) for j in range(n_traj)])                 # (traj, T-1)
    z0 = Z[:, :-1].reshape(-1, Z.shape[-1])
    c = ((Z[:, 1:] - Z[:, :-1]) / dt[..., None]).reshape(-1, Z.shape[-1])
    R = ds.raw_inputs[:, :n_cycles].reshape(n_traj, T, -1)[:, :-1].reshape(n_traj * (T - 1), -1)
    D = dt.reshape(-1, 1)
    P = z0.shape[0]
    z1 = torch.full_like(z0, float("nan"))
    res = {q: torch.full((P,), float("nan"), dtype=torch.float64) for q in ("N", "E", "S")}
    flags = [dict() for _ in range(n_traj)]
    error = [None] * n_traj
    t0 = time.process_time()

    def step(idx):
        zn, info = arm.step(z0[idx], c[idx], R[idx], D[idx], h_steps, track=True)
        zn = zn.detach()
        v0 = lf.evaluate(z0[idx], R[idx], grads=False)
        v1 = lf.evaluate(zn, R[idx], grads=False)
        return zn, info, v0, v1

    def store(idx, zn, info, v0, v1):
        ok = torch.isfinite(zn).all(-1)
        z1[idx[ok]] = zn[ok]
        for q in ("N", "E", "S"):
            if info is not None and "Q_" + q in info:
                r = (v1[q] - v0[q] - info["Q_" + q].detach()).to(torch.float64)
                res[q][idx[ok]] = r[ok]
        if info is None:
            return
        for r, p in enumerate(idx.tolist()):
            if not bool(ok[r]):
                continue
            fb = flags[p // (T - 1)]
            for k, v in info.items():
                if v.dtype == torch.bool:
                    fb[k] = fb.get(k, 0) + int(v[r])
            hit = [info[k][r] for k in ro.DEFECT_FLAGS if k in info]
            if hit:
                fb["any_defect"] = fb.get("any_defect", 0) + int(torch.stack(hit).any())

    for a in range(0, P, chunk):
        idx = torch.arange(a, min(a + chunk, P))
        try:
            store(idx, *step(idx))
        except torch.linalg.LinAlgError:
            for p in idx.tolist():
                one = torch.tensor([p])
                try:
                    store(one, *step(one))
                except torch.linalg.LinAlgError as exc:
                    j = p // (T - 1)
                    error[j] = error[j] or f"{type(exc).__name__}: {exc}"[:200]
    cpu = (time.process_time() - t0) / max(n_traj, 1)
    out = []
    Z1 = z1.reshape(n_traj, T - 1, -1)
    for j in range(n_traj):
        X = torch.empty(T, n, dtype=ds.X.dtype)
        X[0] = _decode(ae, std, Z[j, :1])[0]
        X[1:] = _decode(ae, std, Z1[j])                  # NaN rows stay NaN (failed pairs)
        X[~torch.isfinite(X).all(-1)] = float("nan")
        Xc = X.reshape(n_cycles, ns, n)
        finite = torch.isfinite(Xc).reshape(n_cycles, -1).all(-1)
        fc = None if bool(finite.all()) else int(torch.nonzero(~finite)[0]) + 1
        r_j = {q: v.reshape(n_traj, T - 1)[j].clone() for q, v in res.items()
               if not (lf.system != "A" and q == "N")}
        out.append(ro.RolloutResult(Xc, X[-1], finite, fc, cpu, flags[j], r_j, error[j]))
    return out


def projected_floor(ae, std, lf, ds, phys, n_cycles: int, h_steps: int, floors: dict, tau_S: float,
                    seed: int = 0, project: bool = True) -> list:
    results = projected_results(ae, std, lf, ds, n_cycles, h_steps, project=project)
    return evaluate(_arm("U+proj" if project else "U", ae, std, lf), ds, phys, n_cycles, h_steps, floors,
                    tau_S, seed, results=results)


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
    """Rows of the round-trip floor (no step, so no violation counts)."""
    results = []
    for j in range(ds.X.shape[0]):
        X = _round_trip(ae, std, ds, j, n_cycles)
        results.append(ro.RolloutResult(X, X[-1, -1], torch.ones(n_cycles, dtype=torch.bool), None, 0.0))
    return evaluate(_arm("U", ae, std, lf), ds, phys, n_cycles, 1, floors, tau_S, seed, results=results)


def decoder_floor(ae, std, lf, ds, phys, n_cycles: int, h_steps: int, floors: dict, tau_S: float,
                  seed: int = 0) -> dict:
    """Both floors for one autoencoder (with its LatentFunctionals) on one set."""
    return {"U": u_floor(ae, std, lf, ds, phys, n_cycles, floors, tau_S, seed),
            "projected": projected_floor(ae, std, lf, ds, phys, n_cycles, h_steps, floors, tau_S, seed)}


def write_floor(path, both: dict, meta: dict) -> None:
    """One results file per autoencoder and set, beside the arms' (rows tagged by floor)."""
    rows = [dict(r, floor=k) for k in ("U", "projected") for r in both[k]]
    write_results(path, rows, dict(meta, kind="decoder_floor"))
