"""Materials and bed constants for System A (PREREG_P3B_draft.md, Global Constraints of plan 1).

The isotherms are the three-term fits to Hanikel et al. 2019 (verify/materials/fit_isotherms.json,
method in docs/materials_3B.md) with concentration-based heats from the measured step shift.
The bed and transport properties follow Paper 1 (ai-mof-cof-dynamics v2.0.1): Ruthven's axial
dispersion and Glueckauf's LDF coefficient from the pellet, with the material's own isotherm.
"""
from __future__ import annotations

import json
import os

import numpy as np

from p3b import _paper1  # noqa: F401  (puts Paper 1 on sys.path)
from p3b.isotherms import MultiTermIsotherm, R, from_fit, p_sat
from gen_parametric_dataset import D_M_WATER, EPS_PELLET, TORTUOSITY  # Paper 1
from solver_fd import AdsorptionPhysicsConfig  # Paper 1

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIT_PATH = os.path.join(_ROOT, "verify", "materials", "fit_isotherms.json")

BED = dict(eps_t=0.39, rho_p=1050.0, d_p=3.25e-3, L=0.10, D_in=0.05, k_z=0.10, h_w=10.0)
DH_C = {"MOF-303": -51.3e3, "Al-fumarate": -48.5e3}


def load_isotherm(name: str) -> MultiTermIsotherm:
    """The fitted three-term isotherm of a 3B material; KeyError for any other name."""
    if name not in DH_C:
        raise KeyError(f"unknown material {name!r}; known: {sorted(DH_C)}")
    with open(FIT_PATH) as f:
        fits = json.load(f)
    return from_fit(fits[name], DH_C[name])


def c_in_from_rh(rh: float, T: float) -> float:
    """Water vapour concentration (mol m^-3) at relative humidity rh and temperature T."""
    return float(rh * p_sat(T) / (R * T))


def bed_physics(v: float, T: float) -> AdsorptionPhysicsConfig:
    """Paper 1's physics object with the 3B bed, superficial velocity v and inlet/wall at T.

    k_LDF depends on the isotherm and the feed, so it is left as NaN here: set it with
    k_ldf(), and a forgotten call fails loudly instead of using Paper 1's default.
    """
    p = AdsorptionPhysicsConfig()
    for key, val in BED.items():
        setattr(p, key, val)
    p.v = float(v)
    p.T_in = p.T_w = float(T)
    p.D_L = 0.7 * 2.5e-5 + 0.5 * p.d_p * (p.v / p.eps_t)  # Ruthven, as in Paper 1's build_physics
    p.k_LDF = float("nan")
    return p


def _q_feed(iso: MultiTermIsotherm, c_in: float, T: float) -> float:
    return float(np.asarray(iso.q_star(np.array([c_in]), np.array([T]))).ravel()[0])


def k_ldf(iso: MultiTermIsotherm, phys, c_in: float, T: float) -> float:
    """Glueckauf LDF coefficient (s^-1) from the pellet, with the chord slope K = q*(c_in)/c_in."""
    K = _q_feed(iso, c_in, T) / c_in
    return 15.0 * EPS_PELLET * (D_M_WATER / TORTUOSITY) / ((phys.d_p / 2) ** 2 * phys.rho_p * K)


def t_stoich(iso: MultiTermIsotherm, phys, c_in: float, T: float) -> float:
    """Stoichiometric breakthrough time (s): ((1-eps) rho_p q* + eps c_in) L / (v c_in)."""
    cap = (1 - phys.eps_t) * phys.rho_p * _q_feed(iso, c_in, T) + phys.eps_t * c_in
    return cap * phys.L / (phys.v * c_in)
