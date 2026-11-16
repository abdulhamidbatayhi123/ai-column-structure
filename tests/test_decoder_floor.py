"""The decoder-only floor (PREREG Section 6, revision R1): epsilon_W and e with no learned dynamics.

- U-type floor: the stored reference states encoded and decoded (what U's released amount can reach
  with perfect latent dynamics);
- projected floor: the encoded reference path, interval by interval, with the balance projection of
  the projected arms applied at every RK4 step (what the projected arms can reach).

Both are scored by `evaluate`, the arms' own metric code. The projected floor steps the constant field
c_i = (enc(x_{i+1}) - enc(x_i)) / dt_i with the production `Arm.step` / `rollout_batch`; RK4 is exact for
a constant field, so with the projection off it reproduces the round trip.
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


def test_projected_floor_tracks_balances_and_has_the_evaluate_schema(case):
    ds, std, phys = case
    ae = Identity()
    lf = _lf(ae, std, phys)
    rows = fl.projected_floor(ae, std, lf, ds, phys, n_cycles=2, h_steps=1, floors=FLOORS, tau_S=TAU)
    for r in rows:
        assert {"trajectory", "seed", "e", "eps", "failed_cycle", "violations", "flags", "cpu_s",
                "W_pred", "W_ref", "violations_method"} <= set(r)
        assert r["violations_method"] == "RK4 stages"
        assert all(isinstance(v, int) for v in r["violations"].values())
    # with an exact autoencoder the projection pulls the encoded reference onto the discrete balance:
    # the water and energy residuals of the projected path are at the projection's tolerance
    res = fl.projected_rollouts(ae, std, lf, ds, n_cycles=2, h_steps=1)
    for r in res:
        assert set(r.residuals) == {"N", "E", "S"}
        assert float(r.residuals["N"].abs().max()) < 1e-8 * float(ds.W_ref.abs().max())


def test_projected_floor_batched_equals_one_trajectory_at_a_time(case):
    ds, std, phys = case
    ae = Affine(ds.X.shape[-1], 12)
    lf = _lf(ae, std, phys)
    a = fl.projected_rollouts(ae, std, lf, ds, n_cycles=2, h_steps=1, batch=3)
    b = fl.projected_rollouts(ae, std, lf, ds, n_cycles=2, h_steps=1, batch=1)
    for x, y in zip(a, b):
        assert x.failed_cycle == y.failed_cycle
        assert torch.allclose(x.X, y.X, rtol=1e-9, atol=1e-12, equal_nan=True)


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
