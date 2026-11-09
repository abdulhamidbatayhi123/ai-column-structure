"""Plan 3, Task 8: evaluation of one trained arm on a test set, and the results file."""
import json

import pytest
import torch
from torch import nn

from p3b.eval.evaluate import evaluate, write_results
from p3b.generate import generate_trajectory
from p3b.latent.data import Standardiser, load_set
from p3b.latent.functionals import LatentFunctionals
from p3b.latent.physics import ColumnPhysics
from p3b.simulate_a import setup_a
from p3b.train.trainer import Arm

F64 = torch.float64
NC, NS = 8, 40
POINTS = [dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=0, n_cycles=2),
          dict(RH_ads=0.30, T_ads=298.15, T_des=343.15, v=0.15, alpha=0.9, beta=0.7, index=1, n_cycles=2)]


class Identity(nn.Module):
    def encode(self, x):
        return x

    def decode(self, z):
        return z


class Still(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1, dtype=F64))

    def forward(self, z, feat):
        return torch.zeros_like(z)


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    d = tmp_path_factory.mktemp("ev")
    paths = []
    for i, pt in enumerate(POINTS):
        p = d / f"A{i}.npz"
        assert generate_trajectory("A", pt, "MOF-303", N_z=32, N_c=NC, path=p, n_snap=NS).ok
        paths.append(p)
    ds = load_set(paths)
    std = Standardiser.fit(ds.X, NC, 3, 2, features=ds.features)
    phys = ColumnPhysics(setup_a(POINTS[0], "MOF-303", NC)[0])
    lf = LatentFunctionals(Identity().decode, std, phys, "A", NC)
    arm = Arm("U", Still(), Identity(), std, lf)
    return ds, phys, arm


def test_one_row_per_trajectory_with_every_metric(case):
    ds, phys, arm = case
    rows = evaluate(arm, ds, phys, n_cycles=2, h_steps=1, floors={"N": 1e-9, "E": 1e-9}, tau_S=1e-9, seed=0)
    assert len(rows) == 2
    for r in rows:
        assert {"trajectory", "seed", "e", "eps", "failed_cycle", "violations", "flags", "cpu_s"} <= set(r)
        assert len(r["e"]) == 2 and len(r["eps"]) == 2
        assert all(0 <= v <= 0.5 for v in r["e"] + r["eps"])            # capped
        assert {"N", "E", "S"} <= set(r["violations"])
        assert all(isinstance(v, int) for v in r["violations"].values())


def test_a_still_model_violates_the_water_balance(case):
    ds, phys, arm = case
    rows = evaluate(arm, ds, phys, n_cycles=1, h_steps=1, floors={"N": 1e-9, "E": 1e-9}, tau_S=1e-9, seed=0)
    assert rows[0]["violations"]["N"] > 0                             # dN/dt = 0 while P_N != 0


def test_results_file_round_trips(case, tmp_path):
    ds, phys, arm = case
    rows = evaluate(arm, ds, phys, n_cycles=1, h_steps=1, floors={"N": 1.0, "E": 1.0}, tau_S=1.0, seed=3)
    meta = {"arm": "U", "system": "A", "material": "MOF-303", "params": 1, "pilot": {"h_steps": 1}}
    p = tmp_path / "res.json"
    write_results(p, rows, meta)
    back = json.loads(p.read_text())
    assert back["rows"] == rows and back["meta"]["arm"] == "U"
    assert {"commit", "paper1_sha256"} <= set(back["meta"])
