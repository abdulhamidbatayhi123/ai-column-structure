"""Fixes from the whole-branch review of Plan 3 (one test per finding)."""
import itertools
import json
import math

import numpy as np
import pytest
import torch
from torch import nn

from p3b.eval.evaluate import evaluate
from p3b.eval.rollout import rollout
from p3b.generate import generate_trajectory
from p3b.latent.arms_structured import SDynamics
from p3b.latent.arms_unstructured import UDynamics
from p3b.latent.data import Standardiser, load_set
from p3b.latent.fno import FNO1d
from p3b.latent.functionals import LatentFunctionals
from p3b.latent.physics import ColumnPhysics
from p3b.simulate_a import setup_a
from p3b.stats.tests import signed_rank
from p3b.stats.words import compose_headline, h3b_test, h6_ratio, variance_decomposition
from p3b.train.trainer import Arm, TrainConfig, train

F64 = torch.float64
NC, NS = 8, 40
POINT_A = dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=0, n_cycles=2)


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    d = tmp_path_factory.mktemp("rf3")
    paths = []
    for i, pt in enumerate((POINT_A, dict(POINT_A, RH_ads=0.3, index=1))):
        p = d / f"A{i}.npz"
        assert generate_trajectory("A", pt, "MOF-303", N_z=32, N_c=NC, path=p, n_snap=NS).ok
        paths.append(p)
    return load_set(paths)


class Identity(nn.Module):
    def encode(self, x):
        return x

    def decode(self, z):
        return z


def _std(ds):
    return Standardiser.fit(ds.X, NC, 3, 2, features=ds.features)


def _phys(ds):
    return ColumnPhysics(setup_a(POINT_A, "MOF-303", NC)[0])


# ---- Critical 1 (extended): runtime errors and a non-finite latent state end the rollout -------
def test_rollout_records_runtime_errors_and_nonfinite_latents(ds):
    std = _std(ds)

    class Raiser(nn.Module):
        def forward(self, z, feat):
            raise RuntimeError("solver blew up")

    with pytest.raises(RuntimeError, match="blew up"):        # plan 4 review: only divergence ends a
        rollout(Arm("U", Raiser(), Identity(), std, None), ds, 0, 1, 1)   # rollout; other errors surface

    class NanZ(nn.Module):
        def forward(self, z, feat):
            return torch.full_like(z, float("nan"))

    class FiniteDecoder(Identity):
        def decode(self, z):
            return torch.nan_to_num(z, nan=0.0)

    r = rollout(Arm("U", NanZ(), FiniteDecoder(), std, None), ds, 0, 1, 1)
    assert r.failed_cycle == 1


# ---- Important 1: a non-finite gradient is never applied ---------------------------------------
def test_nonfinite_gradients_are_skipped(ds):
    std = _std(ds)
    torch.manual_seed(0)
    dyn = UDynamics(NC * 3 + 2, 4, 8, time_scale=100.0)
    nn.init.normal_(dyn.net[-1].weight, std=0.1)
    for p in dyn.parameters():
        p.register_hook(lambda g: torch.full_like(g, float("nan")))
    before = {k: v.clone() for k, v in dyn.state_dict().items()}
    h = train(Arm("U", dyn, Identity(), std, None), ds, TrainConfig(lr=1e-3, iterations=2, seed=0, n_windows=2, n_steps=2))
    assert h["skipped_grad"] == 2
    assert all(torch.equal(before[k], v) for k, v in dyn.state_dict().items())


# ---- Important 3: the time scale is saved with the model -----------------------------------------
@pytest.mark.parametrize("make", [lambda ts: UDynamics(4, 2, 8, time_scale=ts)])
def test_time_scale_round_trips_through_the_state_dict(make):
    a = make(850.0)
    b = make(1.0)
    b.load_state_dict(a.state_dict())
    assert float(b.time_scale) == 850.0


def test_s_time_scale_round_trips(ds):
    std = _std(ds)
    lf = LatentFunctionals(Identity().decode, std, _phys(ds), "A", NC)
    a = SDynamics(NC * 3 + 2, 4, lf, kappa=1.0, width=8, time_scale=850.0)
    b = SDynamics(NC * 3 + 2, 4, lf, kappa=1.0, width=8, time_scale=1.0)
    b.load_state_dict(a.state_dict())
    assert float(b.time_scale) == 850.0


# ---- Important 4: rollouts never build gradient graphs -----------------------------------------
def test_rollout_runs_without_gradient_graphs(ds):
    std = _std(ds)
    seen = []

    class Probe(nn.Module):
        def forward(self, z, feat, raw=None):
            seen.append(torch.is_grad_enabled())
            return torch.zeros_like(z)

    lf = LatentFunctionals(Identity().decode, std, _phys(ds), "A", NC)
    rollout(Arm("U", Probe(), Identity(), std, lf), ds, 0, 1, 1, track=True)
    assert seen and not any(seen)


# ---- Important 5: violations are None when not computed; FNO gets the trapezoid; no silent zeros -
def test_violations_none_when_not_tracked_and_fno_counted(ds):
    std = _std(ds)
    phys = _phys(ds)

    class Still(nn.Module):
        def forward(self, z, feat):
            return torch.zeros_like(z)

    with pytest.raises(ValueError, match="lf"):
        evaluate(Arm("U", Still(), Identity(), std, None), ds, phys, 1, 1, {"N": 1, "E": 1}, 1.0, 0)
    fno = FNO1d(NC, 3, 2, 4, modes=4, width=4)
    rows = evaluate(Arm("FNO", fno, None, std, None), ds, phys, 1, 1, {"N": 1e-12, "E": 1e-12}, 1e-12, 0)
    assert all(isinstance(r["violations"]["E"], int) for r in rows)
    assert rows[0]["violations_method"] == "trapezoid on own snapshots"


# ---- Important 6: unknown arm kinds raise -----------------------------------------------------------
def test_unknown_arm_kind_raises(ds):
    with pytest.raises(ValueError, match="arm"):
        Arm("U-proj", nn.Linear(2, 2), Identity(), _std(ds), None)


# ---- Important 7: rows carry what the diagnostics and audits need ----------------------------------
def test_rows_keep_fields_releases_and_uncapped_values(ds):
    std = _std(ds)
    lf = LatentFunctionals(Identity().decode, std, _phys(ds), "A", NC)

    class Still(nn.Module):
        def forward(self, z, feat):
            return torch.zeros_like(z)

    rows = evaluate(Arm("U", Still(), Identity(), std, lf), ds, _phys(ds), 2, 1, {"N": 1, "E": 1}, 1.0, 0)
    r = rows[0]
    assert {"e_fields", "W_pred", "W_ref", "W_ref_exact", "e_uncapped", "eps_uncapped"} <= set(r)
    assert len(r["e_fields"]) == 2 and len(r["e_fields"][0]) == 3
    json.dumps(r)


# ---- Important 8: the small-n signed-rank test is exact under ties ---------------------------------
def test_signed_rank_exact_with_ties_at_small_n():
    p, _ = signed_rank(-np.ones(8), alternative="less")
    assert p == pytest.approx(1 / 256, rel=1e-12)
    x = np.array([-1.0, -1.0, 0.5, -2.0, -2.0, 0.0, -0.5, 1.0, -3.0, -0.5])
    nz = x[x != 0]
    ranks = (st_rank := __import__("scipy").stats.rankdata(np.abs(x)))[x != 0]
    T = ranks[nz > 0].sum()
    allT = [sum(r for r, s in zip(ranks, signs) if s) for signs in itertools.product([0, 1], repeat=nz.size)]
    exact = np.mean(np.array(allT) <= T + 1e-12)
    p, n0 = signed_rank(x, alternative="less")
    assert p == pytest.approx(exact, rel=1e-12) and n0 == 1


# ---- Important 9: the composed headline, validation, H3b and H6 wrappers, variance decomposition ---
def test_compose_headline_fills_qualifiers_and_demotes():
    text = compose_headline(row="2", carrying=["H1"], h1="SUPPORTED", h2="REFUTED (REVERSED)",
                            attribution=["the constraint"], consistency={"H1": ("CONSISTENT", "INCONSISTENT")},
                            coarse="unresolved against the coarse solver", h4="EQUIVALENT", m=1.10)
    assert "helps in one setting" in text and "stability" in text
    assert "+ no data efficiency (REVERSED)" in text
    assert "the constraint" in text and "unresolved against the coarse solver" in text
    assert "(accuracy equivalent within ±10 %)" in text


def test_headline_validates_its_inputs():
    from p3b.stats.words import headline
    with pytest.raises(ValueError):
        headline(False, "REFUTED", "REFUTED", "EQUIVALENT", True)
    with pytest.raises(ValueError):
        headline(False, "SUPPORTD", "REFUTED", "EQUIVALENT", "pass")


def test_h3b_and_h6_wrappers_and_variance_decomposition():
    rng = np.random.default_rng(0)
    drift = rng.normal(size=48)
    err = drift + rng.normal(scale=0.5, size=48)
    rho, p = h3b_test(drift, err, n_perm=2000, seed=0)
    assert rho > 0.5 and p < 0.01
    assert h6_ratio(np.array([0.0, 0.02]), np.array([0.01, 0.0])).tolist() == pytest.approx([0.5, 3.0])
    v = rng.normal(size=(48, 5)) + rng.normal(size=(1, 5))
    dec = variance_decomposition(v)
    assert {"trajectory", "seed", "residual"} <= set(dec) and all(x >= 0 for x in dec.values())


# ---- Ruling (reviewer): an exact-port floor beside the trapezoid floor ------------------------------
def test_reference_floor_reports_the_exact_port_floor(ds):
    from p3b.eval.violations import reference_floor
    fl = reference_floor(ds, _phys(ds))
    assert {"N", "E", "N_exact", "E_exact"} <= set(fl)
    assert all(math.isfinite(fl[k]) and fl[k] >= 0 for k in fl)
