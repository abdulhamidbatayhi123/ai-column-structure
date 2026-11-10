"""S's cost without changing its arithmetic (2026-11-10, after the Kaggle timing: S 27 x U).

Three reuses, each pinned bitwise against the old path:
- the isotherm terms computed once for E^, N^ and S^ (ColumnPhysics.values);
- the stage values of the projected RK4 step taken from the dynamics' own evaluations;
- no second Jacobian at the point where the Gauss-Newton projection has converged.
"""
import torch

from p3b.latent.arms_structured import projected_rk4_step, structured_step
from p3b.latent.integrate import balance_project
from test_arm_s import NC, D, K, _module, _rand

F64 = torch.float64


def test_column_physics_values_equal_the_separate_functionals_bitwise():
    s, _ = _module()
    phys = s.lf.phys
    torch.manual_seed(5)
    ch = torch.stack([torch.randn(4, NC, dtype=F64) * 0.1, -1 + 0.1 * torch.randn(4, NC, dtype=F64),
                      5.74 + 0.01 * torch.randn(4, NC, dtype=F64)], -1)
    v = phys.values(ch)
    assert torch.equal(v["E"], phys.E(ch)) and torch.equal(v["N"], phys.N(ch)) and torch.equal(v["S"], phys.S(ch))


def _count(lf):
    calls = {"n": 0}
    orig = lf.evaluate

    def ev(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    lf.evaluate = ev
    return calls


def test_structured_step_reuses_the_stage_values_bitwise_with_fewer_evaluations():
    s, raw = _module()
    torch.manual_seed(6)
    z, feat = 0.1 * _rand(4, D), _rand(4, K)
    h = torch.full((4, 1), 0.5, dtype=F64)
    with torch.no_grad():
        calls = _count(s.lf)
        z_new, info = structured_step(s, z, feat, raw, h)
        n_new = calls["n"]
        calls["n"] = 0
        z_ref, info_ref = projected_rk4_step(lambda x: s(x, feat, raw)[0], z, raw, h, s.lf)
        n_ref = calls["n"]
    assert torch.equal(z_new, z_ref)
    for k in ("Q_E", "Q_N", "Q_S", "E0", "N0", "S0"):
        assert torch.equal(info[k], info_ref[k]), k
    assert n_new == n_ref - 4


def test_structured_step_gradients_are_unchanged_by_the_reuse():
    s, raw = _module()
    torch.manual_seed(7)
    z, feat = 0.1 * _rand(4, D), _rand(4, K)
    h = torch.full((4, 1), 0.5, dtype=F64)
    params = list(s.G.parameters()) + list(s.A_net.parameters())
    z_new, _ = structured_step(s, z, feat, raw, h)
    g_new = torch.autograd.grad(z_new.square().sum(), params)
    z_ref, _ = projected_rk4_step(lambda x: s(x, feat, raw)[0], z, raw, h, s.lf)
    g_ref = torch.autograd.grad(z_ref.square().sum(), params)
    # the values now share graph nodes with the latent gradients, so autograd sums in another order:
    # equal up to rounding (measured 2.3e-16 relative), not bit for bit
    assert all(torch.allclose(a, b, rtol=1e-12, atol=1e-12 * b.abs().max().item()) for a, b in zip(g_new, g_ref))


def test_balance_project_does_not_recompute_the_jacobian_where_it_converged():
    A = torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]], dtype=F64)
    calls = {"n": 0}

    def cons(x):
        calls["n"] += 1
        return x @ A.T                                  # linear: one Gauss-Newton update converges
    z = torch.tensor([[0.3, -0.2, 0.5]], dtype=F64)
    target = torch.tensor([[1.0, 2.0]], dtype=F64)
    zs, info = balance_project(z, cons, target)
    assert bool(info["converged"].all())
    assert calls["n"] == 2                              # before the update, and at the converged point
