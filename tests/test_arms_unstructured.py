"""Plan 2, Task 5: latent functionals, unstructured arms U, U+F, P and the parameter matcher."""
import pytest
import torch

from p3b.latent.arms_unstructured import (UDynamics, UFDynamics, count_params, match_width, penalty_loss,
                                          penalty_weight)
from p3b.latent.autoencoder import MLPAutoencoder
from p3b.latent.data import Standardiser
from p3b.latent.functionals import LatentFunctionals
from p3b.latent.physics import ColumnPhysics, StorePhysics
from p3b.simulate_a import setup_a
from p3b.simulate_b import T_AMB, make_store

F64 = torch.float64
NC = 8
POINT_A = dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=0, n_cycles=1)


def _functionals_a(d=6):
    torch.manual_seed(0)
    col, sched = setup_a(POINT_A, "MOF-303", NC)
    n = NC * 3 + 2
    # standardiser centred on a plausible state: log c ~ 0, log x ~ -1, log T ~ log 310
    mean = torch.tensor([0.0, -1.0, 5.74] * NC + [0.0, 5.74], dtype=F64)
    std = Standardiser(mean, torch.full((n,), 0.05, dtype=F64))
    ae = MLPAutoencoder(n, d)
    return LatentFunctionals(ae.decode, std, ColumnPhysics(col), "A", NC), sched


def test_latent_functionals_match_the_decoded_physics():
    lf, sched = _functionals_a()
    z = 0.1 * torch.randn(3, 6, dtype=F64)
    c_in, T_in, T_w = sched[0][2]
    raw = torch.tensor([[c_in, T_in, T_w, 0.1]] * 3, dtype=F64)
    out = lf.evaluate(z, raw)
    X = lf.std.decode(lf.decode(z))
    ch, o = X[:, :NC * 3].reshape(3, NC, 3), X[:, NC * 3:]
    assert torch.allclose(out["E"], lf.phys.E(ch), rtol=1e-14)
    assert torch.allclose(out["S"], lf.phys.S(ch), rtol=1e-14)
    P_N, P_E, Phi = lf.phys.ports(ch, o, raw)
    assert torch.allclose(out["P_E"], P_E) and torch.allclose(out["Phi_S"], Phi)
    # latent gradient of S against central differences
    e = torch.zeros(6, dtype=F64)
    e[2] = 1e-6
    fd = (lf.evaluate(z + e, raw, grads=False)["S"] - lf.evaluate(z - e, raw, grads=False)["S"]) / 2e-6
    assert torch.allclose(out["dS"][:, 2], fd, rtol=1e-6)


def test_store_functionals_have_no_water_terms():
    st = make_store(NC)
    n = NC * 2 + 1
    std = Standardiser(torch.full((n,), 320.0, dtype=F64), torch.full((n,), 5.0, dtype=F64))
    lf = LatentFunctionals(MLPAutoencoder(n, 4).decode, std, StorePhysics(st, T_AMB), "B", NC)
    out = lf.evaluate(torch.zeros(2, 4, dtype=F64), torch.tensor([[2e-3, 353.15]] * 2, dtype=F64))
    assert "N" not in out and "dN" not in out and out["dE"].shape == (2, 4)


def test_u_layer_sizes():
    u = UDynamics(32, 4, 128)
    sizes = [m.weight.shape for m in u.modules() if isinstance(m, torch.nn.Linear)]
    assert sizes == [(128, 36), (128, 128), (128, 128), (32, 128)]
    assert u(torch.zeros(5, 32, dtype=F64), torch.zeros(5, 4, dtype=F64)).shape == (5, 32)


def test_uf_input_size():
    assert UFDynamics(32, 4, 128, "A").net[0].in_features == 32 + 4 + 6 + 3 * 32
    assert UFDynamics(16, 4, 128, "B").net[0].in_features == 16 + 4 + 4 + 2 * 16


def test_match_width_hits_the_target_and_refuses_the_impossible():
    target = count_params(UDynamics(32, 4, 150))
    w = match_width(lambda w: UDynamics(32, 4, w), target)
    assert abs(count_params(UDynamics(32, 4, w)) - target) <= 0.05 * target
    with pytest.raises(ValueError):
        match_width(lambda w: UDynamics(32, 4, w), 10)


def test_penalty_weight_is_one_percent_of_the_data_loss():
    assert penalty_weight(2.0, 0.5) == pytest.approx(0.04)
    assert 0.04 * 0.5 == pytest.approx(0.01 * 2.0)


def test_penalty_terms():
    rN = torch.tensor([0.0, 2.0], dtype=F64)
    rE = torch.tensor([1.0, 1.0], dtype=F64)
    sig = torch.tensor([-1.0, 3.0], dtype=F64)     # only negative production is penalised
    p = penalty_loss(rN, rE, sig, scales=(2.0, 1.0, 1.0))
    assert float(p) == pytest.approx(0.5 + 1.0 + 0.5)
    assert float(penalty_loss(None, rE, sig, scales=(None, 1.0, 1.0))) == pytest.approx(1.5)
