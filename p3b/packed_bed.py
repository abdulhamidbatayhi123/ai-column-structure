"""Fluid-to-particle heat exchange and axial conduction in packed beds (revision R3).

One consistent set of coefficients for both systems (Vortmeyer 1989):
  * the particle-to-fluid coefficient of Wakao, Kaguei and Funazkri (1979),
    Nu = h d_p / k = 2 + 1.1 Pr^(1/3) Re^0.6, Re = rho |v| d_p / mu (superficial v), Pr = c mu / k,
    derived together with the fluid's axial dispersion 0.5 Pr Re k (Wakao and Kaguei 1982), which
    both systems already use (p3b.materials.DELTA_AX, ThermalStore.delta_ax);
  * intraparticle conduction lumped into the coefficient (Jeffreson 1972):
    1/h_eff = 1/h + d_p / (10 k_s), with k_s the particle material's own conductivity;
  * the solid surface per bed volume of spheres, a_s = 6 (1 - eps) / d_p.

System B (two temperatures) exchanges h_v(|v|) = a_s h_eff between fluid and solid (ExchangeModel).
System A (one temperature) carries the same fluid-solid resistance in its axial conductivity, by the
equivalence of Vortmeyer and Schaefer (1974) for a moving thermal front:
    k_z = k_e0 + delta_f rho c |v| d_p + Phi (rho c v)^2 / (h_eff a_s),
    Phi = (kappa / (1 + kappa))^2,  kappa = (1 - eps) rho_s c_s / (eps rho c),
the form Tsotsas gives for a one-temperature bed (as restated in Fiers 2009, Eq. 1.48).

Each model has constant fluid properties, so the transport properties are taken at the state its
constant density and heat capacity describe: air at 293.15 K (Paper 1's rho_g = 1.2 kg/m3) and water at
318.15 K (rho_f = 990 kg/m3, c_f = 4180 J/kg/K), recorded by verify/r3/properties.py (Lemmon and
Jacobsen 2004; IAPWS 2008 and 2011; CoolProp 8.0.0). Primary sources and their open restatements:
hub projects/ai-column-structure/T9_r3_equations.md.
"""
from __future__ import annotations

AIR_A = {"mu": 1.8205675178515367e-05, "k": 0.025873828302933142}      # Pa s, W/m/K at 293.15 K
WATER_B = {"mu": 0.0005957693051508003, "k": 0.6347834493539902}       # Pa s, W/m/K at 318.15 K
# System B's rock is granite (density 2.67 g/cm3, the listed rock closest to the store's 2650 kg/m3):
# 2.75 W/m/K at room temperature, from the diffusivity and thermal inertia of Janza (1975) as tabulated
# by Robertson (1988), USGS Open-File Report 88-441, Table 13, p. 90 (K = I sqrt(alpha)); dense granites
# span about 2.1 to 4.2 W/m/K (ibid., Figs. 3-4). Granite's porosity is well under 1 %, so immersion in
# water hardly changes it.
K_ROCK = 2.75
# System A's MOF pellets: no conductivity has been measured for MOF-303 in any form, nor for shaped
# aluminium fumarate. The one measured aluminium fumarate value, 0.12 W/m/K (Gkaniatsou et al. 2021,
# Nano Energy 87, 106224, SI Table S2, transient source), is used for both materials; shaped-MOF
# measurements span 0.07 to 0.16 W/m/K (hub T9 sourcing record).
K_PELLET = 0.12


def nusselt_wakao(re: float, pr: float) -> float:
    """Wakao, Kaguei and Funazkri (1979): particle Nusselt number (2 in still fluid)."""
    return 2.0 + 1.1 * pr ** (1.0 / 3.0) * re ** 0.6


def particle_h(speed: float, d_p: float, rho: float, c: float, mu: float, k: float) -> float:
    """Fluid-to-particle coefficient (W/m2/K) at superficial speed |speed|."""
    return nusselt_wakao(rho * abs(speed) * d_p / mu, c * mu / k) * k / d_p


def jeffreson(h: float, d_p: float, k_s: float | None) -> float:
    """h with the particle's internal conduction resistance in series (Jeffreson 1972); k_s None: h."""
    return h if k_s is None else 1.0 / (1.0 / h + d_p / (10.0 * k_s))


def specific_area(eps: float, d_p: float) -> float:
    """Particle surface per bed volume (1/m) of spheres of diameter d_p at void fraction eps."""
    return 6.0 * (1.0 - eps) / d_p


class ExchangeModel:
    """h_v(speed) = a_s h_eff (W/m3/K) for a two-temperature bed; non-negative for every speed, so the
    store's exchange block stays positive semi-definite. `spec` records every parameter."""
    __name__ = "wakao1979_jeffreson1972"

    def __init__(self, d_p: float, eps: float, rho: float, c: float, mu: float, k: float, k_s: float | None):
        self.d_p, self.eps, self.rho, self.c, self.mu, self.k, self.k_s = d_p, eps, rho, c, mu, k, k_s

    @property
    def spec(self) -> dict:
        return {"name": self.__name__, "d_p": self.d_p, "eps": self.eps, "rho": self.rho, "c": self.c,
                "mu": self.mu, "k": self.k, "k_s": self.k_s}

    def __call__(self, speed: float) -> float:
        h = jeffreson(particle_h(speed, self.d_p, self.rho, self.c, self.mu, self.k), self.d_p, self.k_s)
        return specific_area(self.eps, self.d_p) * h


def one_temperature_kz(v: float, k_e0: float, eps: float, d_p: float, rho: float, c: float, mu: float, k: float,
                       rho_c_solid: float, k_s: float | None, delta_f: float) -> float:
    """Axial conductivity (W/m/K) of a one-temperature bed: stagnant + fluid dispersion + the
    Vortmeyer-Schaefer fluid-solid term. rho_c_solid = (1 - eps) rho_s c_s per bed volume."""
    rho_c = rho * c
    kappa = rho_c_solid / (eps * rho_c)
    phi = (kappa / (1.0 + kappa)) ** 2
    h = jeffreson(particle_h(v, d_p, rho, c, mu, k), d_p, k_s)
    return k_e0 + delta_f * rho_c * abs(v) * d_p + phi * (rho_c * v) ** 2 / (h * specific_area(eps, d_p))


def kz_spec() -> dict:
    """System A's k_z model and its parameters, for the constants fingerprint (p3b.gates)."""
    from p3b import materials as mat
    return {"name": "vortmeyer_schaefer1974_wakao1979", "mu": AIR_A["mu"], "k": AIR_A["k"], "k_s": K_PELLET,
            "delta_f": mat.DELTA_AX}
