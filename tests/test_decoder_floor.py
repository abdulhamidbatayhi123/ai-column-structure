"""The decoder-only floor (PREREG Section 6, revision R1): epsilon_W and e with no learned dynamics.

- U-type floor: the stored reference states encoded and decoded (what U's released amount can reach
  with perfect latent dynamics);
- projected floor: per snapshot interval i, independently, one projected step (the production U+proj
  step, h_steps RK4 substeps each followed by the balance projection) from enc(x_i) with the constant
  field c_i = (enc(x_{i+1}) - enc(x_i)) / dt_i; the decoded result is scored at snapshot i+1. No state is
  carried between intervals, and the interval after the last stored snapshot is not stepped.

Both are scored by `evaluate`, the arms' own metric code. RK4 is exact for a constant field, so with the
projection off the projected floor reproduces the round trip.
"""
import json

import pytest
import torch
from torch import nn

from p3b.eval import floor as fl
from p3b.generate import generate_trajectory
from p3b.latent.data import Standardiser, load_set
from p3b.latent.functionals import LatentFunctionals
from p3b.latent.physics import ColumnPhysics
from p3b.simulate_a import setup_a

F64 = torch.float64
NC, NS = 8, 40
POINTS = [dict(RH_ads=0.30 + 0.08 * i, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.7 + 0.1 * i, beta=0.6,
               index=i, n_cycles=2) for i in range(3)]
FLOORS, TAU = {"N": 1e-9, "E": 1e-9}, 1e-9


class Identity(nn.Module):
    def encode(self, x):
        return x

    def decode(self, z):
        return z


class Affine(nn.Module):
    """An invertible-ish but lossy toy autoencoder: a fixed rank-reducing linear map, so the round trip
    is not exact and the floor is not zero."""

    def __init__(self, n, d):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        q, _ = torch.linalg.qr(torch.randn(n, d, generator=g, dtype=F64))
        self.Q = q

    def encode(self, x):
        return x @ self.Q

    def decode(self, z):
        return z @ self.Q.T


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    d = tmp_path_factory.mktemp("floor")
    paths = []
    for i, pt in enumerate(POINTS):
        p = d / f"t{i}.npz"
        assert generate_trajectory("A", pt, "MOF-303", N_z=32, N_c=NC, path=p, n_snap=NS).ok
        paths.append(p)
    ds = load_set(paths)
    std = Standardiser.fit(ds.X, NC, 3, 2, features=ds.features)
    phys = ColumnPhysics(setup_a(POINTS[0], "MOF-303", NC)[0])
    return ds, std, phys


def _lf(ae, std, phys):
    return LatentFunctionals(ae.decode, std, phys, "A", NC)


def test_identity_autoencoder_has_a_zero_u_floor(case):
    ds, std, phys = case
    ae = Identity()
    rows = fl.u_floor(ae, std, _lf(ae, std, phys), ds, phys, n_cycles=2, floors=FLOORS, tau_S=TAU)
    assert len(rows) == 3
    for r in rows:
        assert r["failed_cycle"] is None
        assert max(r["e"]) < 1e-12 and max(r["eps"]) < 1e-12


def test_u_floor_is_the_round_trip_scored_by_evaluate(case):
    ds, std, phys = case
    ae = Affine(ds.X.shape[-1], 12)
    rows = fl.u_floor(ae, std, _lf(ae, std, phys), ds, phys, n_cycles=2, floors=FLOORS, tau_S=TAU)
    assert max(max(r["e"]) for r in rows) > 1e-6                       # lossy: a non-zero floor
    # the same rows as evaluate() on the round-trip states
    for j, r in enumerate(rows):
        X = ds.X[j, :2]
        rt = std.decode(ae.decode(ae.encode(std.encode(X))))
        assert torch.allclose(torch.tensor(r["W_pred"], dtype=F64),
                              torch.tensor(fl._release_of(rt, ds, j, 2, phys), dtype=F64), rtol=1e-12, atol=0)


def test_projection_off_reproduces_the_u_floor(case):
    ds, std, phys = case
    ae = Affine(ds.X.shape[-1], 12)
    lf = _lf(ae, std, phys)
    u = fl.u_floor(ae, std, lf, ds, phys, n_cycles=2, floors=FLOORS, tau_S=TAU)
    o = fl.projected_floor(ae, std, lf, ds, phys, n_cycles=2, h_steps=2, floors=FLOORS, tau_S=TAU,
                           project=False)
    for a, b in zip(u, o):
        assert b["failed_cycle"] is None
        assert torch.allclose(torch.tensor(a["e"]), torch.tensor(b["e"]), rtol=1e-9, atol=1e-13)
        assert torch.allclose(torch.tensor(a["W_pred"]), torch.tensor(b["W_pred"]), rtol=1e-9, atol=0)


def _stay_rows(ds, phys, lf, ae, std, n_cycles=2):
    """A baseline with zero increments: each scored state is the previous stored snapshot."""
    from dataclasses import replace  # noqa: F401
    from p3b.eval import rollout as ro
    from p3b.eval.evaluate import evaluate
    from p3b.train.trainer import Arm
    res = []
    for j in range(ds.X.shape[0]):
        X = ds.X[j, :n_cycles].reshape(-1, ds.X.shape[-1])
        Y = torch.cat([X[:1], X[:-1]]).reshape(ds.X[j, :n_cycles].shape)
        res.append(ro.RolloutResult(Y, Y[-1, -1], torch.ones(n_cycles, dtype=torch.bool), None, 0.0))
    return evaluate(Arm("U", None, ae, std, lf), ds, phys, n_cycles, 1, FLOORS, TAU, 0, results=res)


def test_projected_floor_tracks_balances_and_has_the_evaluate_schema(case):
    ds, std, phys = case
    ae = Identity()
    lf = _lf(ae, std, phys)
    rows = fl.projected_floor(ae, std, lf, ds, phys, n_cycles=2, h_steps=1, floors=FLOORS, tau_S=TAU)
    for r in rows:
        assert {"trajectory", "seed", "e", "eps", "failed_cycle", "violations", "flags", "cpu_s",
                "W_pred", "W_ref", "violations_method"} <= set(r)
        assert r["violations_method"] == "RK4 stages"
        assert r["failed_cycle"] is None
        assert all(isinstance(v, int) for v in r["violations"].values())
    # the increments carry the reference: with an exact autoencoder and the projection off, every scored
    # state is the stored one (zeroed increments would score each snapshot as the previous one)
    off = fl.projected_floor(ae, std, lf, ds, phys, n_cycles=2, h_steps=1, floors=FLOORS, tau_S=TAU, project=False)
    assert max(max(r["e"]) for r in off) < 1e-12 and max(max(r["eps"]) for r in off) < 1e-12
    # with the projection on, the floor is the projection's own correction of the straight latent path
    # (at this test's coarse 40 snapshots per cycle a few per cent), well below a model that does not move
    stay = _stay_rows(ds, phys, lf, ae, std)
    e_floor = max(max(r["e"]) for r in rows)
    e_stay = max(max(r["e"]) for r in stay)
    eps_floor = max(max(r["eps"]) for r in rows)
    eps_stay = max(max(r["eps"]) for r in stay)
    assert e_floor < 0.5 * e_stay and eps_floor < 0.5 * eps_stay, (e_floor, e_stay, eps_floor, eps_stay)
    # each step meets its own discrete balance from its anchor (the projection's tolerance)
    res = fl.projected_results(ae, std, lf, ds, n_cycles=2, h_steps=1)
    for r in res:
        assert set(r.residuals) == {"N", "E", "S"}
        assert float(r.residuals["N"].abs().max()) < 1e-8 * float(ds.W_ref.abs().max())


def test_no_closing_interval_is_stepped_or_scored(case):
    """n_cycles = the stored cycles (2 x 40 snapshots): there is no next snapshot after the last one, so
    the residuals (and the violation counts) cover the T - 1 = 79 intervals between stored snapshots;
    with one cycle of the two, still T - 1 = 39 (the closing interval is never stepped)."""
    ds, std, phys = case
    ae = Identity()
    lf = _lf(ae, std, phys)
    for n_cycles, n_int in ((2, 2 * NS - 1), (1, NS - 1)):
        for r in fl.projected_results(ae, std, lf, ds, n_cycles=n_cycles, h_steps=1):
            assert all(len(v) == n_int for v in r.residuals.values())
            assert bool(torch.isfinite(r.X).all())


def test_projected_floor_is_re_anchored_at_every_interval(case):
    """A lossy autoencoder with the projection on: each scored state depends only on enc(x_i) and
    enc(x_{i+1}), so perturbing trajectory 0's first snapshot changes its scored snapshots 0 and 1 only,
    and no other trajectory."""
    from dataclasses import replace
    ds, std, phys = case
    ae = Affine(ds.X.shape[-1], 12)
    lf = _lf(ae, std, phys)
    a = fl.projected_results(ae, std, lf, ds, n_cycles=2, h_steps=1)
    X2 = ds.X.clone()
    X2[0, 0, 0] = X2[0, 0, 0] * 1.01
    b = fl.projected_results(ae, std, lf, replace(ds, X=X2), n_cycles=2, h_steps=1)
    xa, xb = a[0].X.reshape(-1, ds.X.shape[-1]), b[0].X.reshape(-1, ds.X.shape[-1])
    assert not torch.allclose(xa[1], xb[1], rtol=1e-9, atol=0)
    assert torch.allclose(xa[2:], xb[2:], rtol=1e-12, atol=1e-14)
    for j in (1, 2):
        assert torch.allclose(a[j].X, b[j].X, rtol=1e-12, atol=1e-14)


def test_projected_floor_batched_equals_one_pair_at_a_time(case):
    ds, std, phys = case
    ae = Affine(ds.X.shape[-1], 12)
    lf = _lf(ae, std, phys)
    a = fl.projected_results(ae, std, lf, ds, n_cycles=2, h_steps=1)
    b = fl.projected_results(ae, std, lf, ds, n_cycles=2, h_steps=1, chunk=1)
    # a residual is a difference of totals of size |N|, |E|, |S|, so batched and one-at-a-time agree to
    # round-off of those totals, not of the residual itself
    with torch.no_grad():
        Z = ae.encode(std.encode(ds.X[:, :2].reshape(-1, ds.X.shape[-1])))
        v = lf.evaluate(Z, ds.raw_inputs[:, :2].reshape(Z.shape[0], -1), grads=False)
    scale = {q: float(v[q].abs().max()) for q in ("N", "E", "S")}
    for x, y in zip(a, b):
        assert x.failed_cycle == y.failed_cycle
        assert torch.allclose(x.X, y.X, rtol=1e-9, atol=1e-12, equal_nan=True)
        for q in x.residuals:
            assert torch.allclose(x.residuals[q], y.residuals[q], rtol=0, atol=1e-12 * scale[q], equal_nan=True)


def test_floor_through_the_pilot_builders_and_the_results_file(tmp_path):
    """The real pilot autoencoder and S's LatentFunctionals (pattern of test_eval_rollout_batch)."""
    import importlib.util
    import pathlib
    from p3b.pilot import pieces
    from p3b.train.cells import _physics
    root = pathlib.Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("time_arms", root / "scripts" / "time_arms.py")
    ta = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ta)
    ta.make_data(str(tmp_path), 32, 8, 40)
    q = {"d": 4, "ae_epochs": 30, "ae_width": 16, "width": 8, "n_windows": 1, "n_steps": 2, "ae_balance_weight": 0.0}
    key = ta.KEY
    train = load_set(pieces.files(str(tmp_path), key, "train"))
    out = tmp_path / "ae"
    out.mkdir()
    st = {"d": {key: 4}}
    pieces.piece_ae(str(tmp_path), key, 0, st, str(out), q)
    s_arm = pieces.make_arm("S", key, 0, st, train, str(out), q)
    phys = _physics("A", "MOF-303", train)
    both = fl.decoder_floor(s_arm.ae, s_arm.std, s_arm.lf, train, phys, n_cycles=2, h_steps=2,
                            floors=FLOORS, tau_S=TAU, seed=0)
    assert set(both) == {"U", "projected"}
    assert len(both["U"]) == len(both["projected"]) == train.X.shape[0]
    for r in both["projected"]:
        assert r["failed_cycle"] is not None or set(r["violations"]) == {"N", "E", "S"}
    p = tmp_path / "floor.json"
    fl.write_floor(p, both, {"system": "A", "material": "MOF-303", "set": "train"})
    back = json.loads(p.read_text())
    assert back["meta"]["kind"] == "decoder_floor"
    assert {r["floor"] for r in back["rows"]} == {"U", "projected"}
    assert len(back["rows"]) == 2 * train.X.shape[0]
