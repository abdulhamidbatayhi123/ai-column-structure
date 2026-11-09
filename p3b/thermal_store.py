"""System B: a packed-bed thermocline store, two-temperature model (design verification only).

A liquid (water, incompressible, constant heat capacity) flows through a packed bed of rock.
The fluid and the solid have their own temperatures (Schumann's two-phase model) and exchange
heat locally. The bed is charged with hot fluid from z = 0 and discharged with cold fluid from
z = L (the flow reverses). There is no sorption and no mass transfer: the only conserved
quantity besides energy is nothing, so the structure is energy plus entropy.

State per cell, per unit bed volume: u_f (fluid energy), u_s (solid energy), with
    u_f = C_f (T_f - T0),  C_f = eps rho_f c_f;     u_s = C_s (T_s - T0),  C_s = (1-eps) rho_s c_s.
Entropy per cell: C_f ln(T_f/T0) + C_s ln(T_s/T0) (strictly concave in (u_f, u_s)).

Mechanisms, each with non-negative entropy production:
  * convection of fluid energy, first-order upwind in the current flow direction
    (tanks in series; Bregman production of the concave stream entropy), ports at both ends;
  * fluid-solid exchange in each cell, W = h_v (T_f - T_s): an Onsager flux
    L (1/T_s - 1/T_f) with L = h_v T_f T_s >= 0;
  * axial conduction in the fluid (effective, incl. dispersion) and in the solid, each
    J = -k dT/dz = L Delta(1/T) with L = k T_i T_{i+1}/dz >= 0;
  * heat loss through the wall from the fluid to the ambient bath (a port).

Optional (defaults reproduce the design-verification model): `d_p` adds the flow-dependent axial
dispersion of the fluid, k_f(v) = k_f + delta_ax rho_f c_f |v| d_p; `compensate=True` lowers it by
the upwind truncation term, k_f,eff = max(0, k_f(v) - rho_f c_f |v| dz/2) (the hybrid scheme).
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import bmat, diags

T0 = 293.15


class ThermalStore:
    def __init__(self, N, L=1.0, eps=0.38, rho_f=990.0, c_f=4180.0, rho_s=2650.0, c_s=830.0,
                 h_v=2.0e4, k_f=1.5, k_s=0.8, h_wall=2.0, D=0.3, d_p=None, delta_ax=0.5, compensate=False):
        self.N, self.L, self.dz = N, L, L / N
        self.params = dict(L=L, eps=eps, rho_f=rho_f, c_f=c_f, rho_s=rho_s, c_s=c_s, h_v=h_v, k_f=k_f, k_s=k_s,
                           h_wall=h_wall, D=D, d_p=d_p, delta_ax=delta_ax, compensate=bool(compensate))
        self.Cf = eps * rho_f * c_f           # J/m3/K per bed volume
        self.Cs = (1 - eps) * rho_s * c_s
        self.rhoc = rho_f * c_f               # per m3 of flowing fluid
        self.h_v, self.k_f, self.k_s = h_v, k_f, k_s
        self.h_a = 4.0 * h_wall / D           # wall, W/m3/K
        self.d_p, self.delta_ax, self.compensate = d_p, delta_ax, bool(compensate)

    def k_f_phys(self, v):
        """Fluid axial conductivity incl. flow-dependent dispersion (W/m/K)."""
        return self.k_f + (self.delta_ax * self.rhoc * abs(v) * self.d_p if self.d_p else 0.0)

    def k_f_eff(self, v):
        """The coefficient the scheme uses: k_f_phys, lowered by the upwind term if compensated."""
        k = self.k_f_phys(v)
        return max(0.0, k - self.rhoc * abs(v) * self.dz / 2) if self.compensate else k

    def split(self, y):
        return y[:self.N], y[self.N:]

    def temps(self, y):
        uf, us = self.split(y)
        return T0 + uf / self.Cf, T0 + us / self.Cs

    def initial_state(self, T):
        return np.concatenate([np.full(self.N, self.Cf * (T - T0)), np.full(self.N, self.Cs * (T - T0))])

    def fluxes(self, y, inputs):
        """inputs = (v, T_in, T_amb); v > 0 flows from z = 0, v < 0 from z = L (feed enters there)."""
        v, T_in, T_amb = inputs
        Tf, Ts = self.temps(y)
        g_in = self.rhoc * (T_in - T0)
        g = self.rhoc * (Tf - T0)
        if v >= 0:
            g_up = np.concatenate(([g_in], g[:-1]))
        else:
            g_up = np.concatenate((g[1:], [g_in]))
        conv = abs(v) / self.dz * (g_up - g)
        W = self.h_v * (Tf - Ts)
        Jf = -self.k_f_eff(v) * (Tf[1:] - Tf[:-1]) / self.dz
        Js = -self.k_s * (Ts[1:] - Ts[:-1]) / self.dz
        Wwall = self.h_a * (Tf - T_amb)
        return dict(Tf=Tf, Ts=Ts, g=g, g_up=g_up, conv=conv, W=W, Jf=Jf, Js=Js, Wwall=Wwall)

    def rhs(self, t, y, inputs):
        f = self.fluxes(y, inputs)
        dz = self.dz

        def div(J):
            out = np.zeros(self.N)
            out[:-1] -= J / dz
            out[1:] += J / dz
            return out

        duf = f["conv"] - f["W"] + div(f["Jf"]) - f["Wwall"]
        dus = f["W"] + div(f["Js"])
        return np.concatenate([duf, dus])

    def sparsity(self):
        band = diags([1.0, 1.0, 1.0], [-1, 0, 1], shape=(self.N, self.N))
        return bmat([[band, band], [band, band]]).tocsr()

    def stream_entropy(self, T):
        return self.rhoc * np.log(T / T0)

    def budget(self, y, inputs):
        v, T_in, T_amb = inputs
        f = self.fluxes(y, inputs)
        duf, dus = self.split(self.rhs(0.0, y, inputs))
        Tf, Ts, dz = f["Tf"], f["Ts"], self.dz
        dE = np.sum(duf + dus) * dz
        T_out = Tf[-1] if v >= 0 else Tf[0]
        portE = abs(v) * self.rhoc * (T_in - T_out) - np.sum(f["Wwall"]) * dz
        dS = np.sum(duf / Tf + dus / Ts) * dz
        portS = abs(v) * (self.stream_entropy(T_in) - self.stream_entropy(T_out)) - np.sum(f["Wwall"]) / T_amb * dz
        T_up = (np.concatenate(([T_in], Tf[:-1])) if v >= 0 else np.concatenate((Tf[1:], [T_in])))
        sig_adv = abs(v) / dz * ((f["g_up"] - f["g"]) / Tf - (self.stream_entropy(T_up) - self.stream_entropy(Tf)))
        sig_ex = f["W"] * (1 / Ts - 1 / Tf)
        sig_cf = f["Jf"] * (1 / Tf[1:] - 1 / Tf[:-1]) / dz
        sig_cs = f["Js"] * (1 / Ts[1:] - 1 / Ts[:-1]) / dz
        sig_wall = f["Wwall"] * (1 / T_amb - 1 / Tf)
        sigma = (np.sum(sig_adv) + np.sum(sig_ex) + np.sum(sig_cf) + np.sum(sig_cs) + np.sum(sig_wall)) * dz
        scale = np.sum(np.abs(duf / Tf) + np.abs(dus / Ts)) * dz
        return dict(dE=dE, portE=portE, dS=dS, portS=portS, sigma=sigma, scale=scale,
                    sig=dict(adv=sig_adv, exchange=sig_ex, cond_f=sig_cf, cond_s=sig_cs, wall=sig_wall))

    def integrate(self, y0, t_span, inputs, n_out, rtol=1e-8, atol=1e-6):
        sol = solve_ivp(self.rhs, t_span, y0, method="BDF", args=(inputs,), t_eval=np.linspace(*t_span, n_out),
                        jac_sparsity=self.sparsity(), rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.t, sol.y
