"""Plan 2, Task 4: RK4 with stage quadrature and the Gauss-Newton balance projection (formulation 5.4)."""
import pytest
import torch

from p3b.latent.integrate import balance_project, rk4_step, stage_quadrature

F64 = torch.float64


def test_rk4_is_fourth_order():
    A = torch.tensor([[0.0, 1.0], [-4.0, -0.3]], dtype=F64)
    z0 = torch.tensor([[1.0, 0.0]], dtype=F64)
    exact = (torch.linalg.matrix_exp(A * 1.0) @ z0.T).T

    def err(n):
        z, h = z0, 1.0 / n
        for _ in range(n):
            z, _ = rk4_step(lambda x: x @ A.T, z, h)
        return float((z - exact).norm())

    assert 14.0 < err(20) / err(40) < 18.0


def test_stages_are_the_rk4_points():
    z0 = torch.zeros(1, 1, dtype=F64)
    _, stages = rk4_step(lambda x: torch.ones_like(x), z0, 0.5)
    assert [float(s[0]) for s in stages] == [0.0, 0.25, 0.25, 0.5]


def test_stage_quadrature_is_exact_for_cubics():
    h = 0.7
    z0 = torch.zeros(1, 1, dtype=F64)
    _, stages = rk4_step(lambda x: torch.ones_like(x), z0, h)       # z(t) = t
    q = stage_quadrature([s[..., 0] ** 3 for s in stages], h)
    assert float(q) == pytest.approx(h ** 4 / 4, rel=1e-14)


def test_projection_restores_linear_constraints_in_one_step():
    torch.manual_seed(0)
    a, b = torch.randn(6, dtype=F64), torch.randn(6, dtype=F64)
    z = torch.randn(3, 6, dtype=F64)
    target = torch.randn(3, 2, dtype=F64)
    zp, info = balance_project(z, lambda x: torch.stack([x @ a, x @ b], -1), target, iters=1)
    assert torch.allclose(torch.stack([zp @ a, zp @ b], -1), target, atol=1e-12)
    # minimum-norm: the change lies in span(a, b)
    dz = zp - z
    basis = torch.stack([a, b], 1)
    resid = dz - (dz @ basis) @ torch.linalg.pinv(basis)
    assert float(resid.abs().max()) < 1e-12
    assert not info["rank_deficient"].any()


def test_projection_restores_nonlinear_constraints():
    torch.manual_seed(1)
    z = torch.randn(4, 8, dtype=F64)
    cons = lambda x: torch.stack([(x ** 2).sum(-1), torch.sin(x).sum(-1)], -1)  # noqa: E731
    target = cons(z) + 1e-3 * torch.randn(4, 2, dtype=F64)
    zp, info = balance_project(z, cons, target, iters=3)
    assert float((cons(zp) - target).abs().max()) < 1e-12
    assert float(info["residual"].max()) < 1e-12


def test_rank_deficient_constraints_are_handled_and_flagged():
    a = torch.arange(1.0, 6.0, dtype=F64)
    z = torch.zeros(2, 5, dtype=F64)
    cons = lambda x: torch.stack([x @ a, 2 * (x @ a)], -1)  # noqa: E731
    target = torch.tensor([[1.0, 2.0], [3.0, 6.0]], dtype=F64)
    zp, info = balance_project(z, cons, target, iters=2)
    assert torch.isfinite(zp).all()
    assert torch.allclose(cons(zp), target, atol=1e-12)
    assert info["rank_deficient"].all()


def test_projection_is_differentiable():
    z = torch.randn(2, 4, dtype=F64, requires_grad=True)
    cons = lambda x: (x ** 2).sum(-1, keepdim=True)  # noqa: E731
    zp, _ = balance_project(z, cons, torch.ones(2, 1, dtype=F64), iters=2, create_graph=True)
    (g,) = torch.autograd.grad(zp.sum(), z)
    assert torch.isfinite(g).all()
