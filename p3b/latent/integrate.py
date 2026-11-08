"""RK4 with stage quadrature and the Gauss-Newton balance projection (formulation 5.4; Plan 2, Task 4).

Inputs are piecewise constant per snapshot interval, so one RK4 step sees one input: the caller
closes the dynamics over it, f(z) -> dz/dt. The port integrals of a step are taken with the same
four RK4 stages (`stage_quadrature`), which is the discrete balance every arm is measured with
(PREREG Section 5). `balance_project` moves z, with the minimum-norm Gauss-Newton update, back
onto the affine-in-time constraints N^(z) = N^_prev + Q[P_N] and E^(z) = E^_prev + Q[P_E].
States are batched: z has shape (batch, d).
"""
from __future__ import annotations

import torch

RANK_TOL = 1e-8


def rk4_step(f, z, h):
    """One classical RK4 step. Returns (z_new, [z1, z2, z3, z4]), the four stage states."""
    z1 = z
    k1 = f(z1)
    z2 = z + 0.5 * h * k1
    k2 = f(z2)
    z3 = z + 0.5 * h * k2
    k3 = f(z3)
    z4 = z + h * k3
    k4 = f(z4)
    return z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), [z1, z2, z3, z4]


def stage_quadrature(g, h):
    """The RK4 weights applied to a quantity evaluated at the four stages: h/6 (g1 + 2 g2 + 2 g3 + g4)."""
    return (h / 6.0) * (g[0] + 2 * g[1] + 2 * g[2] + g[3])


def _jacobian(cons, z, create_graph):
    with torch.enable_grad():
        zz = z if z.requires_grad else z.detach().requires_grad_(True)
        c = cons(zz)
        rows = [torch.autograd.grad(c[..., i].sum(), zz, create_graph=create_graph, retain_graph=True)[0]
                for i in range(c.shape[-1])]
    return c, torch.stack(rows, dim=-2)          # (batch, m), (batch, m, d)


def balance_project(z, cons, target, iters: int = 20, rank_tol: float = RANK_TOL, create_graph: bool = False,
                    tol: float = 1e-12):
    """Gauss-Newton with the pseudo-inverse: z <- z - pinv(J) (cons(z) - target), until every
    |cons(z) - target| <= tol |target| or after `iters` updates.

    Returns (z, info) with info["residual"] the largest |cons(z) - target| per sample after the
    last update and info["rank_deficient"] True where J had fewer than m singular values above
    rank_tol times the largest (formulation 5.3, case (i)); such steps are counted, never hidden.
    """
    deficient = torch.zeros(z.shape[0], dtype=torch.bool)
    for _ in range(iters):
        c, J = _jacobian(cons, z, create_graph)
        if bool(((c - target).abs() <= tol * target.abs()).all()):
            break
        s = torch.linalg.svdvals(J)
        deficient |= (s > rank_tol * s[..., :1]).sum(-1) < J.shape[-2]
        dz = (torch.linalg.pinv(J, rtol=rank_tol) @ (c - target).unsqueeze(-1)).squeeze(-1)
        z = z - dz if create_graph else (z - dz).detach()
    with torch.no_grad():
        resid = (cons(z) - target).abs().amax(-1)
    return z, {"residual": resid, "rank_deficient": deficient}
