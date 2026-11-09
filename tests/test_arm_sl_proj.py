"""Plan 2, Task 7: S-L (learned potentials) and the projected rollouts U+proj, U+F+proj."""
import copy

import torch

from p3b.latent.arms_structured import ProjectedRollout, SLDynamics, projected_rk4_step
from p3b.latent.arms_unstructured import UDynamics
from p3b.latent.autoencoder import MLPAutoencoder
from p3b.latent.data import Standardiser
from p3b.latent.functionals import LatentFunctionals
from p3b.latent.physics import ColumnPhysics
from p3b.simulate_a import setup_a

F64 = torch.float64
NC, D, K = 8, 6, 4
POINT_A = dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=0, n_cycles=1)


def _lf():
    torch.manual_seed(4)
    col, sched = setup_a(POINT_A, "MOF-303", NC)
    n = NC * 3 + 2
    mean = torch.tensor([0.0, -1.0, 5.74] * NC + [0.0, 5.74], dtype=F64)
    std = Standardiser(mean, torch.full((n,), 0.05, dtype=F64))
    lf = LatentFunctionals(MLPAutoencoder(n, D).decode, std, ColumnPhysics(col), "A", NC)
    c_in, T_in, T_w = sched[0][2]
    return lf, torch.tensor([[c_in, T_in, T_w, 0.1]] * 3, dtype=F64)


def test_sl_water_balance_and_learned_energy():
    lf, raw = _lf()
    sl = SLDynamics(D, K, lf, kappa=1e-10)
    z, feat = 0.1 * torch.randn(3, D, dtype=F64), torch.randn(3, K, dtype=F64)
    dz, parts = sl(z, feat, raw)
    t = lf.evaluate(z, raw)
    assert torch.allclose((t["dN"] * dz).sum(-1), t["P_N"], rtol=1e-8)        # physical water balance
    dE_th, dS_th = sl.potential_grads(z)
    M = parts["metriplectic"]
    assert torch.allclose((dE_th * M).sum(-1), torch.zeros(3, dtype=F64), atol=1e-10)   # conserves E_theta
    assert torch.allclose((t["dN"] * M).sum(-1), torch.zeros(3, dtype=F64), atol=1e-10)
    assert torch.all((dS_th * M).sum(-1) >= -1e-12)                          # produces S_theta
    sizes = [m.weight.shape for m in sl.E_net.modules() if isinstance(m, torch.nn.Linear)]
    assert sizes == [(64, D), (64, 64), (1, 64)]


def test_projected_step_restores_both_balances():
    lf, raw = _lf()
    u = UDynamics(D, K, 32)
    feat = torch.randn(3, K, dtype=F64)
    z = 0.1 * torch.randn(3, D, dtype=F64)
    h = 5.0
    z1, info = projected_rk4_step(lambda x: u(x, feat), z, raw, h, lf)
    t0, t1 = lf.evaluate(z, raw, grads=False), lf.evaluate(z1, raw, grads=False)
    assert torch.allclose(t1["N"] - t0["N"], info["Q_N"], rtol=0, atol=1e-10 * float(t0["N"].abs().max()))
    assert torch.allclose(t1["E"] - t0["E"], info["Q_E"], rtol=0, atol=1e-10 * float(t0["E"].abs().max()))


def test_projected_rollout_does_not_retrain():
    lf, raw = _lf()
    u = UDynamics(D, K, 32)
    before = copy.deepcopy(u.state_dict())
    pr = ProjectedRollout(lambda x, f, r: u(x, f), lf)
    z = 0.1 * torch.randn(3, D, dtype=F64)
    feat = torch.randn(3, K, dtype=F64)
    for _ in range(3):
        z, _ = pr.step(z, feat, raw, 2.0)
    assert all(torch.equal(before[k], v) for k, v in u.state_dict().items())
    assert sum(p.numel() for p in pr.parameters()) == 0
