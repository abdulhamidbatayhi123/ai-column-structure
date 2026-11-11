"""Arm C+L: the consistent hybrid scheme at 32 cells with learned non-negative multipliers
(PREREG Section 3; formulation 4.2; Plan 2, Task 10).

System A: multipliers on eps D_L,eff and k_z,eff (per face, from the face mean of the local inputs)
and on k_LDF (per cell). System B: on k_f,eff (per face) and the exchange coefficient h_v (per
cell). Each multiplier is the softplus of an MLP 3 -> 32 -> 32 -> 1 (SiLU) of the cell state; the
last layer starts at weight 0 and bias ln(e - 1), so every multiplier starts at exactly 1 and C+L
starts as C. Scaling a non-negative Onsager coefficient by a non-negative factor keeps every block
positive semi-definite, so C+L is consistent by construction (formulation 4.2); `production`
checks it mechanism by mechanism.

The right-hand sides are torch float64 transcriptions of `p3b.column.Column.rhs` (hybrid scheme)
and `p3b.thermal_store.ThermalStore.rhs`, checked against them to 1e-12. Implicit Euler is solved
by Newton; its gradient comes from one Newton correction with a detached Jacobian at the converged
point, y = y* - (I - h J)^-1 (y* - y0 - h f(y*)), which equals y* in value and gives exactly the
implicit-function-theorem gradient with respect to y0 and the multiplier weights.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from p3b.column import C_REF, R, T0
from p3b.latent.autoencoder import _mlp
from p3b.latent.physics import Q_torch
from p3b.thermal_store import T0 as T0_B

F64 = torch.float64
_INV_SOFTPLUS_1 = math.log(math.e - 1.0)
STALL_TOL = 1e-8     # Newton may stop where it stalls only below this block-relative residual
CHORD_RATE = 0.5     # modified Newton reuses its Jacobian while each iteration at least halves the residual


class Multipliers(nn.Module):
    """n_out independent non-negative multipliers of 3 local inputs; `constant` fixes them (tests)."""

    def __init__(self, n_out: int, n_in: int = 3, width: int = 32, constant: float | None = None):
        super().__init__()
        self.n_out, self.constant = n_out, constant
        self.nets = nn.ModuleList(_mlp([n_in, width, width, 1]) for _ in range(n_out))
        for net in self.nets:
            nn.init.zeros_(net[-1].weight)
            nn.init.constant_(net[-1].bias, _INV_SOFTPLUS_1)

    def forward(self, x):
        if self.constant is not None:
            return torch.full(x.shape[:-1] + (self.n_out,), float(self.constant), dtype=F64)
        return torch.cat([nn.functional.softplus(net(x)) for net in self.nets], dim=-1)


def _div(J, N, dz):
    zero = torch.zeros(J.shape[:-1] + (1,), dtype=J.dtype)
    return (torch.cat([J, zero], -1) * -1 + torch.cat([zero, J], -1)) / dz


PER_POINT = ("v", "D_L", "k_z", "k_LDF")      # the column constants that differ between trajectories


class TorchColumnRHS(nn.Module):
    """Column.rhs (system A, hybrid scheme) with multipliers [eps D_L,eff, k_z,eff, k_LDF]."""

    def __init__(self, col, mult: Multipliers | None = None):
        super().__init__()
        self.col, self.mult = col, mult
        p = col.p
        self.N, self.dz, self.v = col.N, col.dz, float(p.v)
        self.eps, self.A, self.Cs, self.c_w, self.dH = col.eps, col.A, col.Cs, col.c_w, col.dH
        self.rhoC, self.h_a, self.iso = col.rhoC, col.h_a, col.iso
        self.D_L, self.k_z, self.k_LDF, self.compensate = float(p.D_L), float(p.k_z), float(p.k_LDF), col.compensate
        self.q_cap = col.iso.q_capacity

    @classmethod
    def stack(cls, rhs_list):
        """One right-hand side for a batch of trajectories (states (B, 3N), inputs (B, 3)): the
        per-trajectory constants PER_POINT become (B, 1) tensors; the multipliers and every other
        constant must be shared (one material, one grid)."""
        first = rhs_list[0]
        for r in rhs_list[1:]:
            if r.mult is not first.mult:
                raise ValueError("a batch shares one set of multipliers")
            same = all(getattr(r, k) == getattr(first, k)
                       for k in ("N", "dz", "eps", "A", "Cs", "c_w", "dH", "rhoC", "h_a", "compensate", "q_cap"))
            if not same or vars(r.iso) != vars(first.iso):
                raise ValueError("a batch shares one material and one grid")
        out = cls(first.col, first.mult)
        for k in PER_POINT:
            setattr(out, k, torch.tensor([[float(getattr(r, k))] for r in rhs_list], dtype=F64))
        return out

    def e_w(self, T):
        return self.c_w * (T - T0)

    def stream_energy(self, c, T):
        return self.rhoC * (T - T0) + c * self.e_w(T)

    def phi_over_T(self, T):
        return self.e_w(T) / T - self.c_w * torch.log(T / T0)

    def state(self, y):
        N = self.N
        n, a, u = y[..., :N], y[..., N:2 * N], y[..., 2 * N:]
        T = T0 + (u - a * self.dH - (n + a) * self.e_w(torch.tensor(T0, dtype=F64))) / (self.Cs + (n + a) * self.c_w)
        return n, a, u, n / self.eps, a / self.A, T

    def local_inputs(self, c, q, T):
        return torch.stack([torch.log(torch.clamp(c, min=1e-30)), q / self.q_cap, T / 300.0], -1)

    def multipliers(self, c, q, T):
        if self.mult is None:
            one_f, one_c = torch.ones(c.shape[:-1] + (self.N - 1,), dtype=F64), torch.ones_like(c)
            return one_f, one_f, one_c
        x = self.local_inputs(c, q, T)
        n_f = self.N - 1                                  # faces and cells through the networks at once
        m = self.mult(torch.cat([0.5 * (x[..., 1:, :] + x[..., :-1, :]), x], dim=-2))
        mf, mc = m[..., :n_f, :], m[..., n_f:, :]
        return mf[..., 0], mf[..., 1], mc[..., 2]

    def fluxes(self, y, inp):
        c_in, T_in, T_w = inp[..., 0:1], inp[..., 1:2], inp[..., 2:3]
        n, a, u, c, q, T = self.state(y)
        dz, v = self.dz, self.v
        mD, mk, mL = self.multipliers(c, q, T)
        b = self.iso.b0 * torch.exp(-self.dH / (R * T))
        r_ldf = self.A * self.k_LDF * mL * (Q_torch(self.iso, b * torch.clamp(c, min=0.0)) - q)
        epsD = self.eps * self.D_L
        k_eff = self.k_z * torch.ones(c.shape[:-1] + (self.N - 1,), dtype=F64)
        if self.compensate:
            epsD = torch.clamp(torch.as_tensor(epsD - v * dz / 2, dtype=F64), min=0.0)
            c_face = 0.5 * (c[..., 1:] + c[..., :-1])
            k_eff = torch.clamp(self.k_z - (self.rhoC + c_face * self.c_w) * v * dz / 2, min=0.0)
        epsD = epsD * mD
        k_eff = k_eff * mk
        J_n = -epsD * (c[..., 1:] - c[..., :-1]) / dz
        dinvT = 1.0 / T[..., 1:] - 1.0 / T[..., :-1]
        dphi = self.phi_over_T(T[..., 1:]) - self.phi_over_T(T[..., :-1])
        small = dinvT.abs() < 1e-14
        eta = torch.where(small, self.e_w(0.5 * (T[..., 1:] + T[..., :-1])),
                          dphi / torch.where(small, torch.ones_like(dinvT), dinvT))
        J_u = eta * J_n - k_eff * (T[..., 1:] - T[..., :-1]) / dz
        g_n = torch.cat([c_in.expand(c.shape[:-1] + (1,)), c], -1)
        g_u = torch.cat([self.stream_energy(c_in, T_in).expand(c.shape[:-1] + (1,)), self.stream_energy(c, T)], -1)
        W = self.h_a * (T - T_w)
        return dict(c=c, q=q, T=T, r_ldf=r_ldf, J_n=J_n, J_u=J_u, g_n=g_n, g_u=g_u, W=W)

    def forward(self, y, inp):
        f = self.fluxes(y, inp)
        dz, v, N = self.dz, self.v, self.N
        dn = v * (f["g_n"][..., :-1] - f["g_n"][..., 1:]) / dz + _div(f["J_n"], N, dz) - f["r_ldf"]
        du = v * (f["g_u"][..., :-1] - f["g_u"][..., 1:]) / dz + _div(f["J_u"], N, dz) - f["W"]
        return torch.cat([dn, f["r_ldf"], du], -1)

    def _X(self, q):
        lo, hi = torch.full_like(q, -60.0), torch.full_like(q, 60.0)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            below = Q_torch(self.iso, torch.exp(mid)) < q
            lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
        return torch.exp(0.5 * (lo + hi))

    @torch.no_grad()
    def production(self, y, inp) -> dict:
        """Entropy production per mechanism with the learned multipliers: LDF per cell, dispersion
        plus conduction per face (formulation 4.2); each must be >= 0."""
        f = self.fluxes(y, inp)
        c, q, T = f["c"], f["q"], f["T"]
        wn = -(self.phi_over_T(T) + R * torch.log(c / C_REF))
        wa = -(self.phi_over_T(T) + self.dH / T + R * torch.log(self._X(q) / (self.iso.b0 * C_REF)))
        sig_ldf = f["r_ldf"] * (wa - wn)
        sig_face = (f["J_n"] * (wn[1:] - wn[:-1]) + f["J_u"] * (1 / T[1:] - 1 / T[:-1])) / self.dz
        return {"ldf": sig_ldf, "face": sig_face}


class TorchStoreRHS(nn.Module):
    """ThermalStore.rhs (system B) with multipliers [k_f,eff (per face), h_v (per cell)]."""

    def __init__(self, st, mult: Multipliers | None = None):
        super().__init__()
        self.st, self.mult = st, mult
        self.N, self.dz = st.N, st.dz
        self.Cf, self.Cs, self.rhoc, self.h_v, self.k_s, self.h_a = st.Cf, st.Cs, st.rhoc, st.h_v, st.k_s, st.h_a

    def forward(self, y, inp):
        v, T_in, T_amb = float(inp[0]), inp[1], inp[2]
        N, dz = self.N, self.dz
        Tf, Ts = T0_B + y[:N] / self.Cf, T0_B + y[N:] / self.Cs
        if self.mult is None:
            mk, mh = torch.ones(N - 1, dtype=F64), torch.ones(N, dtype=F64)
        else:
            x = torch.stack([Tf / 300.0, Ts / 300.0, Tf - Ts], -1)
            mk, mh = self.mult(0.5 * (x[1:] + x[:-1]))[:, 0], self.mult(x)[:, 1]
        g_in = (self.rhoc * (T_in - T0_B)).reshape(1)
        g = self.rhoc * (Tf - T0_B)
        g_up = torch.cat([g_in, g[:-1]]) if v >= 0 else torch.cat([g[1:], g_in])
        conv = abs(v) / dz * (g_up - g)
        W = self.h_v * mh * (Tf - Ts)
        Jf = -self.st.k_f_eff(v) * mk * (Tf[1:] - Tf[:-1]) / dz
        Js = -self.k_s * (Ts[1:] - Ts[:-1]) / dz
        Wwall = self.h_a * (Tf - T_amb)
        duf = conv - W + _div(Jf, N, dz) - Wwall
        dus = W + _div(Js, N, dz)
        return torch.cat([duf, dus])


def colored_jacobian(f, y, n_cells: int, n_var: int):
    """Dense Jacobian of a right-hand side whose cell i depends only on cells i-1, i, i+1 (every
    variable block), from 3 n_var forward-mode products (columns coloured by variable and cell
    mod 3) instead of n reverse passes."""
    n = y.numel()
    J = torch.zeros(n, n, dtype=y.dtype)
    rows = torch.arange(n)
    row_cell = rows % n_cells
    seeds = torch.zeros(3 * n_var, n, dtype=y.dtype)
    for var in range(n_var):
        for k in range(3):
            seeds[3 * var + k, var * n_cells + torch.arange(k, n_cells, 3)] = 1.0
    JV = torch.func.vmap(lambda s: torch.func.jvp(f, (y,), (s,))[1])(seeds)   # one batched pass
    for var in range(n_var):
        for k in range(3):
            Jv = JV[3 * var + k]
            for off in (-1, 0, 1):
                cell = row_cell + off
                ok = (cell >= 0) & (cell < n_cells) & (cell % 3 == k)
                J[rows[ok], var * n_cells + cell[ok]] = Jv[ok]
    return J


def batch_jacobian(f, y, n_cells: int | None = None):
    """Per-sample Jacobians (B, n, n) of a right-hand side whose sample b depends only on y[b].

    With n_cells, the stencil is used: cell i of every variable depends only on cells i-1, i, i+1, so
    the rows grouped by variable and cell mod 3 never share a column, and 3 n_var reverse passes give
    the Jacobian exactly (9 for system A, instead of n = 96). This matters with learned multipliers in
    the graph, where the vectorised 96-cotangent pass costs ~0.5 s at batch 8. Without n_cells: the
    Jacobian of the batch sum by vectorised reverse mode."""
    if n_cells is None:
        with torch.enable_grad():
            J = torch.autograd.functional.jacobian(lambda x: f(x).sum(0), y.detach(), vectorize=True)
        return J.detach().permute(1, 0, 2)
    B, n = y.shape
    N = n_cells
    n_var = n // N
    idx = torch.arange(n)
    var_of, cell_of = idx // N, idx % N
    J = torch.zeros(B, n, n, dtype=y.dtype)
    with torch.enable_grad():
        x = y.detach().requires_grad_(True)
        out = f(x)
        for var in range(n_var):
            for k in range(3):
                seed = ((var_of == var) & (cell_of % 3 == k)).to(y.dtype).expand(B, n)
                g, = torch.autograd.grad(out, x, seed, retain_graph=True)
                for off in (-1, 0, 1):                  # the one row of the group touching each column
                    r = cell_of + off
                    ok = (r >= 0) & (r < N) & (r % 3 == k)
                    J[:, var * N + r[ok], idx[ok]] = g[:, idx[ok]]
    return J.detach()


def step_jacobian(f, y):
    """Dense Jacobian of f at y by vectorised reverse mode (one batched backward pass), detached and
    usable under no_grad. Newton's Jacobian: on CPU it is about 4x cheaper than `colored_jacobian`,
    whose torch.func.vmap over jvp falls back to Python decompositions (28 against 111 ms at 32 cells,
    system A; the two agree to round-off)."""
    with torch.enable_grad():
        return torch.autograd.functional.jacobian(f, y.detach(), vectorize=True).detach()


def implicit_euler_step(rhs, y0, inp, h: float, tol: float = 1e-10, max_iter: int = 50, substeps: int = 1,
                        chord: bool = True, jac_cache: dict | None = None):
    """y(h) by `substeps` implicit-Euler steps of h/substeps, each solved by Newton with the
    dense Jacobian (`step_jacobian`); differentiable in y0 and the multiplier weights (see module doc).

    Convergence is tested per variable block (n, a, u for system A; u_f, u_s for B), each against
    the larger of its magnitude and its update h f, because the blocks differ by up to 1e13. A Newton solve that does not
    converge raises RuntimeError: a wrong value with a wrong gradient is never returned. A solve that
    stalls (the residual no longer halves) is accepted only below STALL_TOL, the round-off floor of
    some states lying between 1e-12 and 1e-9, and only after an iteration with a fresh Jacobian.

    With chord (the default) Newton is modified: the factorised Jacobian is reused while each
    iteration cuts the residual by CHORD_RATE or more and refreshed otherwise, and with no gradient
    needed the converged point's Jacobian (used only for the gradient correction, which changes the
    value by less than tol) is not formed. The tolerance is the same, so the answer agrees with full
    Newton to tol; chord=False is full Newton.

    With chord, each substep starts from the previous substep's factorised Jacobian, and jac_cache (a
    dict the caller keeps between snapshot intervals) carries it from one call to the next. It is used
    only for the same batch and the same step sizes (to 1e-9, the rounding of stored times); a non-finite residual on a carried Jacobian
    restarts the solve from the start state with a fresh Jacobian."""
    single = y0.dim() == 1
    if single:                                          # one trajectory: a batch of one
        y0, inp = y0.unsqueeze(0), inp.unsqueeze(0)
    B, n = y0.shape
    N = rhs.N
    n_var = n // N
    eye = torch.eye(n, dtype=F64)
    hs = (torch.as_tensor(h, dtype=F64).reshape(-1, 1) / substeps).expand(B, 1)
    y_prev = y0
    params = list(rhs.parameters()) if hasattr(rhs, "parameters") else []
    need_grad = torch.is_grad_enabled() and (y0.requires_grad or any(q.requires_grad for q in params))
    cache = jac_cache if jac_cache is not None else {}
    for _ in range(substeps):
        if single:                                      # any rhs taking one state (system B's too)
            f = lambda x: rhs(x[0], inp[0]).unsqueeze(0)  # noqa: E731
        else:
            f = lambda x: rhs(x, inp)  # noqa: E731
        with torch.no_grad():
            base = y_prev.detach()
            y = base.clone()
            # each block against the larger of its own magnitude and its update h f: in a pre-wetted
            # bed n is ~1e-4 of the feed term, and a tolerance on |n| alone sits below round-off
            scale = torch.maximum(base.reshape(B, n_var, N).abs().amax(-1, keepdim=True),
                                  (hs * f(base)).reshape(B, n_var, N).abs().amax(-1, keepdim=True)).clamp(min=1e-300)
            ok = False
            prev = torch.full((B,), float("inf"), dtype=F64)
            carried = (chord and cache.get("B") == B and "lu" in cache     # same step sizes up to time rounding
                       and torch.allclose(cache["hs"], hs, rtol=1e-9, atol=0.0))
            lu, fresh, restarted = (cache["lu"] if carried else None), False, False
            for _ in range(max_iter):
                F = y - base - hs * f(y)
                rel = (F.reshape(B, n_var, N).abs() / scale).amax((1, 2))
                if not fresh and not restarted and lu is not None and not bool(torch.isfinite(rel).all()):
                    y, lu, restarted = base.clone(), None, True      # a carried Jacobian went astray
                    prev = torch.full((B,), float("inf"), dtype=F64)
                    continue
                done = rel <= tol
                if fresh:                                   # stalled at the round-off floor of this state
                    done = done | ((rel >= 0.5 * prev) & (rel <= STALL_TOL))
                if bool(done.all()):
                    ok = True
                    break
                fresh = lu is None or not chord or bool((rel > CHORD_RATE * prev).any())
                if fresh:
                    lu = torch.linalg.lu_factor(eye - hs.unsqueeze(-1) * batch_jacobian(f, y, N))
                prev = rel
                y = y - torch.linalg.lu_solve(*lu, F.unsqueeze(-1)).squeeze(-1)
            if not ok:
                F = y - base - hs * f(y)
                ok = bool((F.reshape(B, n_var, N).abs() <= tol * scale).all())
            if not ok:
                worst = float((F.reshape(B, n_var, N).abs() / scale).max())
                raise RuntimeError(f"Newton did not converge in {max_iter} iterations "
                                   f"(largest block-relative residual {worst:.3e}, tol {tol:g})")
            if not (need_grad or not chord):
                if lu is not None:                          # converged without one: keep the cache as it is
                    cache.update(B=B, hs=hs.clone(), lu=lu)
                y_prev = y
                continue
            J = batch_jacobian(f, y, N)
            lu_star = torch.linalg.lu_factor(eye - hs.unsqueeze(-1) * J)
            cache.update(B=B, hs=hs.clone(), lu=lu_star)
        F = y - y_prev - hs * f(y)
        y_prev = y - torch.linalg.lu_solve(*lu_star, F.unsqueeze(-1)).squeeze(-1)
    return y_prev.squeeze(0) if single else y_prev


class CPlusL(nn.Module):
    def __init__(self, rhs):
        super().__init__()
        self.rhs = rhs

    def rollout(self, y0, inps, dts, substeps: int = 1):
        """States after each snapshot interval; inps (steps, 3), dts (steps,)."""
        ys, y = [], y0
        for k in range(inps.shape[0]):
            y = implicit_euler_step(self.rhs, y, inps[k], float(dts[k]), substeps=substeps)
            ys.append(y)
        return torch.stack(ys)
