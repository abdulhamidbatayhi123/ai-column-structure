"""Thermodynamically consistent finite-volume column (formulation_3B.md, Sections 3-4).

DESIGN VERIFICATION CODE, not the study's reference solver. It exists to test the
formulation's claims numerically on small grids before anything is frozen:

  * with c_w = 0 it must reproduce Paper 1's solver_fd.py (formulation, Section 3.4);
  * water and energy close against the port fluxes (Section 4.4);
  * every entropy-production term is non-negative (Section 4.2-4.3);
  * dS/dt = port entropy flux + total production (bookkeeping identity).

State per cell, per unit bed volume: n (vapour mol), a (adsorbed mol), u (internal
energy, J). The isotherm is a MultiTermIsotherm (verify/isotherms.py); by default it is Paper 1's
dual-site isotherm, built from Paper 1's parameters, which reproduces isotherm.q_star_np.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import bmat, diags

from p3b.isotherms import from_paper1

R = 8.314
T0 = 298.15          # reference temperature of e_w and s_w
C_REF = 1.0          # c° (mol/m3), reference concentration of the solute entropy


class Column:
    """Geometry, properties and the thermodynamics of Section 3 for one material.

    `phys` is a Paper 1 AdsorptionPhysicsConfig (or anything with the same fields).
    `c_w` is the molar heat capacity of water in both phases (J/mol/K); c_w = 0 is
    Paper 1's model. `iso` is a MultiTermIsotherm; default: Paper 1's dual-site isotherm.

    `compensate=True` is the hybrid scheme (Spalding 1972; Patankar 1980): the upwind truncation
    term is removed from the physical face coefficients, eps*D_L -> max(0, eps*D_L - v dz/2) and,
    per face, k_z -> max(0, k_z - (rho_g C_pg + c_face c_w) v dz/2), with c_face the face mean.
    Central differencing where the cell Peclet number is at most 2, upwind without physical
    diffusion above it; written as upwind plus non-negative diffusion, the consistency proof
    holds unchanged. Default False (Paper 1's plain upwind).
    """

    def __init__(self, phys, N, c_w, iso=None, compensate=False):
        self.p = phys
        self.N = N
        self.dz = phys.L / N
        self.c_w = float(c_w)
        self.eps = phys.eps_t
        self.A = (1.0 - phys.eps_t) * phys.rho_p              # kg solid per m3 bed
        self.Cs = phys.eps_t * phys.rho_g * phys.C_pg + self.A * phys.C_ps
        self.rhoC = phys.rho_g * phys.C_pg                    # carrier, per m3 gas
        self.h_a = 4.0 * phys.h_w / phys.D_in                 # wall, W/m3/K
        self.iso = iso if iso is not None else from_paper1(phys)
        self.dH = self.iso.dH
        self.compensate = bool(compensate)

    # ---- Section 3: thermodynamics -------------------------------------------------
    def e_w(self, T):
        return self.c_w * (T - T0)

    def s_w(self, T):
        return self.c_w * np.log(T / T0)

    def phi_over_T(self, T):
        """phi°(T)/T with phi° = e_w - T s_w."""
        return self.e_w(T) / T - self.s_w(T)

    def temperature(self, n, a, u):
        return T0 + (u - a * self.dH - (n + a) * self.e_w(T0)) / (self.Cs + (n + a) * self.c_w)

    def energy(self, n, a, T):
        return self.Cs * (T - T0) + (n + a) * self.e_w(T) + a * self.dH

    def X_of_q(self, q):
        return self.iso.X(q)

    def Pi_of_x(self, x):
        return self.iso.Pi(x)

    def q_star(self, c, T):
        return self.iso.q_star(c, T)

    def mu_v_over_T(self, c, T):
        return self.phi_over_T(T) + R * np.log(c / C_REF)

    def mu_a_over_T(self, q, T):
        """mu_a/T = phi°/T + dH/T + R ln(X(q)/(b0 c°)), b(T) = b0 exp(-dH/RT)  (Section 3.2)."""
        return self.phi_over_T(T) + self.dH / T + R * np.log(self.X_of_q(q) / (self.iso.b0 * C_REF))

    def s_adsorbed(self, q, T):
        """Entropy of the adsorbed phase per kg solid (Section 3.2)."""
        x = self.X_of_q(q)
        return q * self.s_w(T) - R * (q * np.log(x / (self.iso.b0 * C_REF)) - self.Pi_of_x(x))

    def cell_entropy(self, n, a, u):
        T = self.temperature(n, a, u)
        c = n / self.eps
        q = a / self.A
        s_gas = n * (self.s_w(T) - R * np.log(c / C_REF) + R)
        return self.Cs * np.log(T / T0) + s_gas + self.A * self.s_adsorbed(q, T)

    def stream_energy(self, c, T):
        """Energy carried per m3 of flowing gas."""
        return self.rhoC * (T - T0) + c * self.e_w(T)

    def stream_entropy(self, c, T):
        """Entropy carried per m3 of flowing gas (same functional form as the cell's gas)."""
        return self.rhoC * np.log(T / T0) + c * (self.s_w(T) - R * np.log(c / C_REF) + R)

    # ---- Section 4: dynamics ------------------------------------------------------
    def split(self, y):
        N = self.N
        return y[:N], y[N:2 * N], y[2 * N:]

    def fluxes(self, y, inputs):
        """All fluxes and rates. inputs = (c_in, T_in, T_w)."""
        c_in, T_in, T_w = inputs
        p = self.p
        n, a, u = self.split(y)
        T = self.temperature(n, a, u)
        c = n / self.eps
        q = a / self.A
        dz = self.dz

        # LDF uptake (within cells)
        r_ldf = self.A * p.k_LDF * (self.q_star(c, T) - q)

        # dispersion (exact Fick on c) and conduction, on interior faces i+1/2
        epsD = self.eps * p.D_L
        k_eff = np.full(self.N - 1, float(p.k_z))
        if self.compensate:
            epsD = max(0.0, epsD - p.v * dz / 2)
            c_face = 0.5 * (c[1:] + c[:-1])
            k_eff = np.maximum(0.0, p.k_z - (self.rhoC + c_face * self.c_w) * p.v * dz / 2)
        J_n = -epsD * (c[1:] - c[:-1]) / dz
        # enthalpy carried by the dispersive flux: eta = Delta(phi/T)/Delta(1/T), the
        # discrete-gradient mean of e_w (Gibbs-Helmholtz: d(phi/T)/d(1/T) = e_w), so that
        # X_n + eta X_u = -R Delta ln c exactly; eta -> e_w(T) when Delta T -> 0
        dinvT = 1.0 / T[1:] - 1.0 / T[:-1]
        dphi = self.phi_over_T(T[1:]) - self.phi_over_T(T[:-1])
        small = np.abs(dinvT) < 1e-14
        eta = np.where(small, self.e_w(0.5 * (T[1:] + T[:-1])),
                       dphi / np.where(small, 1.0, dinvT))
        J_u = eta * J_n - k_eff * (T[1:] - T[:-1]) / dz

        # convection (first-order upwind = tanks in series), gas stream per m3
        g_n = np.concatenate(([c_in], c))
        g_u = np.concatenate(([self.stream_energy(c_in, T_in)], self.stream_energy(c, T)))

        # wall exchange, W per m3 bed
        W = self.h_a * (T - T_w)
        return dict(T=T, c=c, q=q, r_ldf=r_ldf, J_n=J_n, J_u=J_u, eta=eta,
                    g_n=g_n, g_u=g_u, W=W, epsD_eff=epsD, k_eff=k_eff)

    def rhs(self, t, y, inputs):
        f = self.fluxes(y, inputs)
        dz, v = self.dz, self.p.v
        N = self.N

        def div(J):  # interior-face flux divergence, zero flux on both end faces
            out = np.zeros(N)
            out[:-1] -= J / dz
            out[1:] += J / dz
            return out

        dn = v * (f["g_n"][:-1] - f["g_n"][1:]) / dz + div(f["J_n"]) - f["r_ldf"]
        da = f["r_ldf"]
        du = v * (f["g_u"][:-1] - f["g_u"][1:]) / dz + div(f["J_u"]) - f["W"]
        return np.concatenate([dn, da, du])

    def sparsity(self):
        N = self.N
        band = diags([1.0, 1.0, 1.0], [-1, 0, 1], shape=(N, N))
        return bmat([[band, band, band]] * 3).tocsr()

    # ---- Section 4.4: balances and entropy production ----------------------------
    def budget(self, y, inputs):
        """Instantaneous rates: (dN/dt, port_N, dE/dt, port_E, dS/dt, port_S, sigma terms)."""
        c_in, T_in, T_w = inputs
        f = self.fluxes(y, inputs)
        dydt = self.rhs(0.0, y, inputs)
        n, a, u = self.split(y)
        dn, da, du = self.split(dydt)
        T, c, q = f["T"], f["c"], f["q"]
        dz, v = self.dz, self.p.v

        wn = -self.mu_v_over_T(c, T)
        wa = -self.mu_a_over_T(q, T)
        wu = 1.0 / T

        dN = np.sum(dn + da) * dz
        portN = v * (c_in - c[-1])
        dE = np.sum(du) * dz
        portE = v * (self.stream_energy(c_in, T_in) - self.stream_energy(c[-1], T[-1])) - np.sum(f["W"]) * dz
        dS = np.sum(wn * dn + wa * da + wu * du) * dz
        portS = v * (self.stream_entropy(c_in, T_in) - self.stream_entropy(c[-1], T[-1])) \
            - np.sum(f["W"]) / T_w * dz

        # production, mechanism by mechanism (each must be >= 0)
        sig_ldf = f["r_ldf"] * (wa - wn)                     # = r (mu_v - mu_a)/T
        Xn = wn[1:] - wn[:-1]
        Xu = wu[1:] - wu[:-1]
        sig_face = (f["J_n"] * Xn + f["J_u"] * Xu) / self.dz  # per m3, dispersion + conduction
        gn_up, gu_up = f["g_n"][:-1], f["g_u"][:-1]
        T_up = np.concatenate(([T_in], T[:-1]))
        s_up = self.stream_entropy(gn_up, T_up)
        s_here = self.stream_entropy(c, T)
        sig_adv = (v / dz) * (wn * (gn_up - c) + wu * (gu_up - f["g_u"][1:]) - (s_up - s_here))
        sig_wall = f["W"] * (1.0 / T_w - 1.0 / T)
        return dict(dN=dN, portN=portN, dE=dE, portE=portE, dS=dS, portS=portS,
                    sig_ldf=sig_ldf, sig_face=sig_face, sig_adv=sig_adv, sig_wall=sig_wall,
                    sigma_total=(np.sum(sig_ldf) + np.sum(sig_adv) + np.sum(sig_wall)) * dz
                    + np.sum(sig_face) * dz)

    # ---- integration ---------------------------------------------------------------
    def initial_state(self, T_init):
        n = np.full(self.N, 1e-12 * self.eps)      # clean bed (tiny positive c for ln c)
        a = np.full(self.N, 1e-12 * self.A)
        T = np.full(self.N, T_init)
        return np.concatenate([n, a, self.energy(n, a, T)])

    def integrate(self, y0, t_span, inputs, n_out, rtol=1e-8, atol=1e-12):
        sol = solve_ivp(self.rhs, t_span, y0, method="BDF", args=(inputs,),
                        t_eval=np.linspace(*t_span, n_out), jac_sparsity=self.sparsity(),
                        rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.t, sol.y
