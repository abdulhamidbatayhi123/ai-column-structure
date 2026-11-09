"""RK4 with stage quadrature and the Gauss-Newton balance projection (formulation 5.4; Plan 2, Task 4).

Inputs are piecewise constant per snapshot interval, so one RK4 step sees one input: the caller
closes the dynamics over it, f(z) -> dz/dt. The port integrals of a step are taken with the same
four RK4 stages (`stage_quadrature`), which is the discrete balance every arm is measured with
(PREREG Section 5). `balance_project` moves z, with the minimum-norm Gauss-Newton update, back
onto the affine-in-time constraints N^(z) = N^_prev + Q[P_N] and E^(z) = E^_prev + Q[P_E].
States are batched: z has shape (batch, d).

Units: the constraints mix J m^-2 and mol m^-2 (|grad E^| / |grad N^| ~ |dH| ~ 5e4), so the rows of
the Jacobian are scaled to unit norm before the rank test and the pseudo-inverse. In exact
arithmetic the minimum-norm step does not change under row scaling; the relative singular-value
tolerance then measures the angle between the constraint gradients, not their units.
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
    """The RK4 weights applied to a quantity evaluated at the four stages: h/6 (g1 + 2 g2 + 2 g3 + g4).
    A per-sample step h of shape (batch, 1) (as used for the states) is applied per sample."""
    if torch.is_tensor(h) and h.dim() == 2 and g[0].dim() == 1:
        h = h[:, 0]
    return (h / 6.0) * (g[0] + 2 * g[1] + 2 * g[2] + g[3])


def _jacobian(cons, z):
    with torch.enable_grad():
        zz = z.detach().requires_grad_(True)
        c = cons(zz)
        rows = [torch.autograd.grad(c[..., i].sum(), zz, retain_graph=True)[0] for i in range(c.shape[-1])]
    return c.detach(), torch.stack(rows, dim=-2)          # (batch, m), (batch, m, d)


def row_scaled(J):
    """(J with unit-norm rows, the row norms (batch, m, 1)); zero rows stay zero."""
    norms = J.norm(dim=-1, keepdim=True)
    return J / torch.where(norms > 0, norms, torch.ones_like(norms)), norms


def rank_deficient(Js, rank_tol: float = RANK_TOL):
    """Fewer than m singular values above rank_tol times the largest (rows already unit-scaled)."""
    with torch.no_grad():
        s = torch.linalg.svdvals(Js)
        return (s > rank_tol * s[..., :1]).sum(-1) < Js.shape[-2]


def balance_project(z, cons, target, iters: int = 20, rank_tol: float = RANK_TOL, tol: float = 1e-12,
                    scale=None, differentiable: bool | None = None):
    """Gauss-Newton with the row-scaled pseudo-inverse, until |cons(z) - target| <= tol * scale for
    every constraint of every sample, or after `iters` updates.

    `scale` (batch, m) is the physical scale of each constraint (the violation-metric
    normalisation, PREREG 5); default |target|. The iterations run without a graph. If
    `differentiable` (default: gradients enabled and z or target requires grad), the returned z
    carries the first-order gradient of the projection at the converged point z*:
        z = z* - J+ (cons(z*) - target) + (I - J+ J)(z_in - z_in.detach()),
    exact for linear constraints; for curved ones the error is the curvature times the distance
    projected. Returns (z, info) with per-sample "residual" (largest |cons - target| / scale),
    "converged" and "rank_deficient" (formulation 5.3, case (i); counted, never hidden).
    """
    if differentiable is None:
        differentiable = torch.is_grad_enabled() and (z.requires_grad or target.requires_grad)
    td = target.detach()
    sc = td.abs() if scale is None else scale.detach().expand_as(td)
    sc = torch.where(sc > 0, sc, torch.full_like(sc, 1e-300))
    zs = z.detach()
    deficient = torch.zeros(z.shape[0], dtype=torch.bool)
    for _ in range(iters):
        c, J = _jacobian(cons, zs)
        if bool(((c - td).abs() <= tol * sc).all()):
            break
        Js, norms = row_scaled(J)
        deficient |= rank_deficient(Js, rank_tol)
        r = (c - td) / torch.where(norms[..., 0] > 0, norms[..., 0], torch.ones_like(norms[..., 0]))
        zs = zs - (torch.linalg.pinv(Js, rtol=rank_tol) @ r.unsqueeze(-1)).squeeze(-1)
    c, J = _jacobian(cons, zs)
    rel = ((c - td).abs() / sc)
    info = {"residual": rel.amax(-1), "converged": (rel <= tol).all(-1), "rank_deficient": deficient}
    if not differentiable:
        return zs, info
    Js, norms = row_scaled(J)
    nz = torch.where(norms[..., 0] > 0, norms[..., 0], torch.ones_like(norms[..., 0]))
    Jp = torch.linalg.pinv(Js, rtol=rank_tol)                                  # (batch, d, m)
    dzin = z - z.detach()
    tangent = dzin - (Jp @ (Js @ dzin.unsqueeze(-1))).squeeze(-1)
    z_out = zs - (Jp @ ((c - target) / nz).unsqueeze(-1)).squeeze(-1) + tangent
    return z_out, info
