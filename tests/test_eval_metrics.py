"""Plan 3, Task 1: cycle metrics (PREREG Section 5)."""
import numpy as np
import pytest
import torch

from p3b import gates
from p3b.eval.metrics import apply_failure, cycle_errors, relative_error, release
from p3b.generate import generate_trajectory
from p3b.latent.data import load_set
from p3b.latent.metrics import field_nrmse
from p3b.latent.physics import ColumnPhysics, StorePhysics
from p3b.simulate_a import setup_a
from p3b.simulate_b import T_AMB, make_store

POINT_A = dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=0, n_cycles=2)
POINT_B = dict(T_hot=353.15, T_cold=298.15, v=2.0e-3, alpha=0.7, beta=0.8, index=0, n_cycles=2)
NC, NS = 8, 200


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("m")
    out = {}
    for sys_, pt, mat in (("A", POINT_A, "MOF-303"), ("B", POINT_B, None)):
        p = d / f"{sys_}.npz"
        assert generate_trajectory(sys_, pt, mat, N_z=32, N_c=NC, path=p, n_snap=NS).ok
        out[sys_] = p
    return out


def test_cycle_errors_zero_and_equal_to_the_gate_metric(files):
    ds = load_set([files["A"]])
    phys = ColumnPhysics(setup_a(POINT_A, "MOF-303", NC)[0])
    X = ds.X[0]
    assert torch.all(cycle_errors(X, X, phys, NC) == 0)
    Y = X.clone()
    Y[..., 2:NC * 3:3] += 0.01                                    # log T + 0.01
    e = cycle_errors(Y, X, phys, NC)
    ch = lambda Z, k: phys.physical_fields(Z[k][:, :NC * 3].reshape(-1, NC, 3)).reshape(NS, -1)  # noqa: E731
    for k in range(2):
        assert float(e[k]) == pytest.approx(float(field_nrmse(ch(Y, k), ch(X, k), NC, 3)), rel=1e-12)


def test_release_by_trapezoid_matches_the_exact_ports(files):
    for sys_, phys in (("A", None), ("B", make_store(NC))):
        ds = load_set([files[sys_]])
        d = np.load(files[sys_])
        W_exact = gates.cycle_release(d)
        X = ds.X[0]
        n_state = X.shape[-1]
        W = release(X[..., n_state - (2 if sys_ == "A" else 1):], ds.raw_inputs[0], ds.t[0], sys_,
                    rhoc=None if sys_ == "A" else phys.rhoc)
        if sys_ == "A":
            assert W.shape == (1,)                      # A: the last cycle needs a closing snapshot
        else:
            assert W.shape == (2,)                      # B: last interval by the left rectangle
        for k in range(W.shape[0]):
            assert float(W[k]) == pytest.approx(float(W_exact[k]), rel=2e-3)


def test_release_with_a_closing_state_covers_every_cycle(files):
    ds = load_set([files["A"]])
    X = ds.X[0]
    W = release(X[..., -2:], ds.raw_inputs[0], ds.t[0], "A", closing=(X[1, 0, -2:], float(ds.t[0, 1, -1]) + 1.0))
    assert W.shape == (2,)


def test_relative_error():
    assert torch.allclose(relative_error(torch.tensor([1.1, 0.9], dtype=torch.float64), torch.tensor([1.0, 1.0], dtype=torch.float64)),
                          torch.tensor([0.1, 0.1], dtype=torch.float64), rtol=1e-12)


def test_failure_capping():
    e = torch.tensor([0.1, 0.2, 0.6, 0.1, 0.1])
    eps = torch.tensor([0.05, 0.1, 0.1, 0.1, 0.1])
    ec, xc, k = apply_failure(e, eps, torch.ones(5, dtype=torch.bool))
    assert k == 3 and ec.tolist() == pytest.approx([0.1, 0.2, 0.5, 0.5, 0.5]) and xc[2:].tolist() == [0.5] * 3
    fin = torch.tensor([True, False, True, True, True])
    ec, xc, k = apply_failure(torch.tensor([0.1, float("nan"), 0.1, 0.1, 0.1]), eps, fin)
    assert k == 2 and torch.isfinite(ec).all() and ec[1:].tolist() == [0.5] * 4
    ec, xc, k = apply_failure(e.clamp(max=0.4), torch.tensor([0.1, 0.7, 0.1, 0.1, 0.1]), torch.ones(5, dtype=torch.bool))
    assert k == 2
    ec, xc, k = apply_failure(torch.full((3,), 0.1), torch.full((3,), 0.1), torch.ones(3, dtype=torch.bool))
    assert k is None and ec.tolist() == pytest.approx([0.1] * 3)
