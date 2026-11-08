"""Plan 3, Task 2: balance and entropy violations with each arm's own stages (PREREG 5)."""
import numpy as np
import pytest
import torch

from p3b.eval.violations import count_violations, normalisers, reference_floor, step_residuals
from p3b.generate import generate_trajectory
from p3b.latent.arms_structured import projected_rk4_step
from p3b.latent.arms_unstructured import UDynamics
from p3b.latent.autoencoder import MLPAutoencoder
from p3b.latent.data import Standardiser, load_set
from p3b.latent.functionals import LatentFunctionals
from p3b.latent.integrate import rk4_step
from p3b.latent.physics import ColumnPhysics
from p3b.simulate_a import setup_a

F64 = torch.float64
NC = 8
POINT_A = dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=0, n_cycles=2)


def _lf():
    torch.manual_seed(4)
    col, sched = setup_a(POINT_A, "MOF-303", NC)
    n = NC * 3 + 2
    mean = torch.tensor([0.0, -1.0, 5.74] * NC + [0.0, 5.74], dtype=F64)
    lf = LatentFunctionals(MLPAutoencoder(n, 6).decode, Standardiser(mean, torch.full((n,), 0.05, dtype=F64)),
                           ColumnPhysics(col), "A", NC)
    c_in, T_in, T_w = sched[0][2]
    return lf, torch.tensor([[c_in, T_in, T_w, 0.1]] * 3, dtype=F64)


def test_projected_step_has_zero_balance_residuals():
    lf, raw = _lf()
    u = UDynamics(6, 4, 16)
    feat = torch.randn(3, 4, dtype=F64)
    z0 = 0.1 * torch.randn(3, 6, dtype=F64)
    with torch.no_grad():
        f = lambda x: u(x, feat)  # noqa: E731
        _, stages = rk4_step(f, z0, 2.0)
        z1, _ = projected_rk4_step(f, z0, raw, 2.0, lf)
        r = step_residuals(lf, z0, z1, stages, 2.0, raw)
        z1u, stages_u = rk4_step(f, z0, 2.0)
        ru = step_residuals(lf, z0, z1u, stages_u, 2.0, raw)
    assert float(r["N"].abs().max()) < 1e-10 * float(lf.evaluate(z0, raw, grads=False)["N"].abs().max())
    assert float(ru["N"].abs().max()) > 1e3 * float(r["N"].abs().max())   # unprojected U violates
    assert set(r) == {"N", "E", "S"}


def test_counts_exceed_only_beyond_ten_times_the_floor():
    r = torch.tensor([0.5, 2.0, 9.9, 10.1, 30.0], dtype=F64)
    assert count_violations(r, scale=torch.ones(5, dtype=F64), floor=1.0) == 2
    assert count_violations(r, scale=torch.full((5,), 2.0, dtype=F64), floor=1.0) == 1


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    p = tmp_path_factory.mktemp("v") / "A.npz"
    assert generate_trajectory("A", POINT_A, "MOF-303", N_z=32, N_c=NC, path=p, n_snap=40).ok
    return load_set([p])


def test_normalisers(ds):
    n = normalisers(ds)
    sched = ds.meta[0]["schedule"]
    feed = POINT_A["v"] * sched[0][2][0] * (sched[0][1] + sched[1][1])
    assert n["feed_water"].shape == (1, 2) and float(n["feed_water"][0, 0]) == pytest.approx(feed, rel=1e-12)
    assert torch.all(n["energy_scale"] > 0)


def test_reference_floor_is_finite_and_positive(ds):
    phys = ColumnPhysics(setup_a(POINT_A, "MOF-303", NC)[0])
    fl = reference_floor(ds, phys)
    assert {"N", "E"} <= set(fl)
    assert all(np.isfinite(fl[k]) and fl[k] > 0 for k in ("N", "E"))
