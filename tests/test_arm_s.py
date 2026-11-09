"""Plan 2, Task 6: arm S and its ablations S-M, S-Pi; every guarantee of formulation 5.3 is a test."""
import pytest
import torch

from p3b.latent.arms_structured import SDynamics, structure_terms
from p3b.latent.autoencoder import MLPAutoencoder
from p3b.latent.data import Standardiser
from p3b.latent.functionals import LatentFunctionals
from p3b.latent.physics import ColumnPhysics
from p3b.simulate_a import setup_a

F64 = torch.float64
NC, D, K = 8, 6, 4
POINT_A = dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=0, n_cycles=1)


def _rand(*s):
    return torch.randn(*s, dtype=F64)


def test_equalities_hold_exactly():
    torch.manual_seed(0)
    B, dS, G, A = _rand(5, D, 2), _rand(5, D), _rand(5, D), _rand(5, D, D)
    t, phi = _rand(5, 2), _rand(5)
    out, info = structure_terms(B, dS, G, A, t, phi, kappa=1e-8)
    lhs = (B * out.unsqueeze(-1)).sum(-2)
    assert torch.allclose(lhs, t, rtol=1e-10, atol=1e-10)


def test_entropy_inequality_up_to_the_counted_deficit():
    torch.manual_seed(1)
    B, dS, G, A = _rand(50, D, 2), _rand(50, D), _rand(50, D), _rand(50, D, D)
    t, phi = _rand(50, 2), 10 * _rand(50)
    out, info = structure_terms(B, dS, G, A, t, phi, kappa=1e-3)
    prod = (dS * out).sum(-1) - phi
    assert torch.all(prod >= -info["residual_deficit"] - 1e-10)
    free = ~info["kappa_binds"]
    assert free.any() and torch.all(prod[free] >= -1e-3 * phi.abs()[free] - 1e-10)


def test_metriplectic_term_conserves_and_produces():
    torch.manual_seed(2)
    B, dS, A = _rand(20, D, 2), _rand(20, D), _rand(20, D, D)
    out, _ = structure_terms(B, dS, torch.zeros(20, D, dtype=F64), A, None, None, kappa=1e-8,
                             port_projection=False)
    assert torch.allclose((B * out.unsqueeze(-1)).sum(-2), torch.zeros(20, 2, dtype=F64), atol=1e-10)
    assert torch.all((dS * out).sum(-1) >= -1e-12)


def test_uniform_state_is_finite_and_counts_kappa():
    B = _rand(3, D, 2)
    dS = (B @ _rand(3, 2, 1)).squeeze(-1)         # grad S in span(B): P grad S = 0 (formulation 5.3 ii)
    out, info = structure_terms(B, dS, _rand(3, D), _rand(3, D, D), _rand(3, 2), _rand(3).abs() + 1, kappa=1e-6)
    assert torch.isfinite(out).all() and info["kappa_binds"].all()


def test_rank_deficient_b_is_finite_and_flagged():
    b = _rand(3, D, 1)
    B = torch.cat([b, 2 * b], -1)                   # parallel grad E and grad N (formulation 5.3 i)
    t = torch.cat([torch.ones(3, 1), 2 * torch.ones(3, 1)], -1).to(F64)
    out, info = structure_terms(B, _rand(3, D), _rand(3, D), _rand(3, D, D), t, _rand(3), kappa=1e-6)
    assert torch.isfinite(out).all() and info["rank_deficient"].all()


def _module(**kw):
    torch.manual_seed(3)
    col, sched = setup_a(POINT_A, "MOF-303", NC)
    n = NC * 3 + 2
    mean = torch.tensor([0.0, -1.0, 5.74] * NC + [0.0, 5.74], dtype=F64)
    std = Standardiser(mean, torch.full((n,), 0.05, dtype=F64))
    lf = LatentFunctionals(MLPAutoencoder(n, D).decode, std, ColumnPhysics(col), "A", NC)
    c_in, T_in, T_w = sched[1][2]
    raw = torch.tensor([[c_in, T_in, T_w, 0.1]] * 4, dtype=F64)
    return SDynamics(D, K, lf, kappa=1e-10, **kw), raw


def test_s_module_meets_the_balances_on_a_decoder():
    s, raw = _module()
    z, feat = 0.1 * _rand(4, D), _rand(4, K)
    dz, info = s(z, feat, raw)
    t = s.lf.evaluate(z, raw)
    assert torch.allclose((t["dE"] * dz).sum(-1), t["P_E"], rtol=1e-8)
    assert torch.allclose((t["dN"] * dz).sum(-1), t["P_N"], rtol=1e-8)
    assert torch.all((t["dS"] * dz).sum(-1) - t["Phi_S"] >= -info["residual_deficit"] - 1e-9 * t["Phi_S"].abs())


def test_parameter_shapes():
    s, _ = _module()
    sizes = [m.weight.shape for m in s.G.modules() if isinstance(m, torch.nn.Linear)]
    assert sizes == [(128, D + K), (128, 128), (128, 128), (D, 128)]
    assert [m.weight.shape for m in s.A_net.modules() if isinstance(m, torch.nn.Linear)] == [(128, D), (D * D, 128)]


def test_ablations():
    s_m, raw = _module(metriplectic=False)
    s_p, _ = _module(port_projection=False)
    z, feat = 0.1 * _rand(4, D), _rand(4, K)
    t = s_p.lf.evaluate(z, raw)
    dz_p, _ = s_p(z, feat, raw)
    assert not torch.allclose((t["dE"] * dz_p).sum(-1), t["P_E"], rtol=1e-3)   # S-Pi: no port projection
    dz_m, _ = s_m(z, feat, raw)
    assert torch.allclose((t["dE"] * dz_m).sum(-1), t["P_E"], rtol=1e-8)      # S-M still projects ports
    assert s_m.A_net is None
