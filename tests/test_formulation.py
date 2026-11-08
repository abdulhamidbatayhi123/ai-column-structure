"""Fast property tests of the consistent column (formulation_3B.md, Sections 3-4).

Run: <Paper 1 venv python> -m pytest -q tests
Each test is seconds. They lock in what verify/run_checks.py showed on 2026-11-03.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "verify"))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "ai-mof-cof-dynamics")))
from consistent_column import Column  # noqa: E402
from isotherms import from_fit, from_paper1  # noqa: E402
import json  # noqa: E402
from gen_parametric_dataset import build_physics, rh_to_conc  # noqa: E402
from solver_fd import generate_breakthrough_data  # noqa: E402

MAT = dict(q_max=24.5, delta_H=-52000.0, step_rh=0.12, isotherm_n=3.0,
           henry_fraction=0.17, d_p=3.25e-3, rho_p=1050.0, eps_t=0.39)
COND = dict(T_in=303.15, rh_feed=0.40, v=0.10)


@pytest.fixture(scope="module")
def phys():
    return build_physics(MAT, COND)


FITS = json.load(open(os.path.join(HERE, "..", "verify", "materials", "fit_isotherms.json")))
DH_C = {"MOF-303": -51.3e3, "Al-fumarate": -48.5e3}   # concentration-based heats, docs/materials_3B.md


def iso_for(name, phys):
    return from_paper1(phys) if name == "paper1" else from_fit(FITS[name], DH_C[name])


def test_paper1_isotherm_reproduced(phys):
    """The two-term MultiTermIsotherm built from Paper 1's parameters equals isotherm.q_star_np."""
    from isotherm import q_star_np
    iso = from_paper1(phys)
    c = np.linspace(0.0, 3.0, 301)
    for T in (293.15, 320.0, 360.0):
        assert np.allclose(iso.q_star(c, np.full_like(c, T)), q_star_np(c, np.full_like(c, T), phys),
                           rtol=1e-12, atol=1e-14)


def random_state(col, rng):
    """A non-equilibrium state with a front-like profile and a temperature bump."""
    N = col.N
    z = np.linspace(0, 1, N)
    c = 10 ** rng.uniform(-3, 0.3) * (0.05 + 1 / (1 + np.exp((z - rng.uniform(0.2, 0.8)) * 30)))
    q = col.p.q_max * rng.uniform(0.02, 0.9) * (0.1 + 1 / (1 + np.exp((z - rng.uniform(0.2, 0.8)) * 20)))
    T = 300 + 25 * np.exp(-((z - rng.uniform(0.2, 0.8)) / 0.1) ** 2) + rng.normal(0, 0.5, N)
    n, a = c * col.eps, q * col.A
    return np.concatenate([n, a, col.energy(n, a, T)])


@pytest.mark.parametrize("c_w,iso_name", [(0.0, "paper1"), (33.6, "paper1"),
                                            (33.6, "MOF-303"), (33.6, "Al-fumarate")])
def test_identities_and_production_signs(phys, c_w, iso_name):
    rng = np.random.default_rng(1)
    col = Column(phys, 40, c_w=c_w, iso=iso_for(iso_name, phys))
    c_in = rh_to_conc(0.4, 303.15)
    for inputs in [(c_in, 303.15, 303.15), (c_in, 353.15, 353.15), (0.2 * c_in, 330.0, 340.0)]:
        for _ in range(5):
            y = random_state(col, rng)
            b = col.budget(y, inputs)
            f = col.fluxes(y, inputs)
            dn, da, du = col.split(col.rhs(0.0, y, inputs))
            scale_S = np.sum(np.abs(col.mu_v_over_T(f["c"], f["T"]) * dn)
                             + np.abs(col.mu_a_over_T(f["q"], f["T"]) * da)
                             + np.abs(du / f["T"])) * col.dz
            assert abs(b["dN"] - b["portN"]) <= 1e-12 * (abs(b["dN"]) + abs(b["portN"]) + 1e-30) + 1e-14
            assert abs(b["dE"] - b["portE"]) <= 1e-11 * (abs(b["dE"]) + abs(b["portE"]))
            assert abs(b["dS"] - b["portS"] - b["sigma_total"]) <= 1e-10 * scale_S
            for key in ("sig_ldf", "sig_face", "sig_adv", "sig_wall"):
                assert np.min(b[key]) >= -1e-12 * (abs(b["sigma_total"]) / col.dz + 1e-30), key


def test_dispersion_is_exact_fick(phys):
    """The Onsager dispersion block reproduces Paper 1's discrete Fick flux exactly."""
    rng = np.random.default_rng(2)
    col = Column(phys, 30, c_w=33.6)
    y = random_state(col, rng)
    f = col.fluxes(y, (1.0, 303.15, 303.15))
    c, T = f["c"], f["T"]
    eta = f["eta"]
    Xn = -(col.mu_v_over_T(c[1:], T[1:]) - col.mu_v_over_T(c[:-1], T[:-1]))
    Xu = 1 / T[1:] - 1 / T[:-1]
    logmean = (c[1:] - c[:-1]) / (np.log(c[1:]) - np.log(c[:-1]))
    L_nn = col.eps * phys.D_L / (8.314 * col.dz) * logmean
    J_onsager = L_nn * (Xn + eta * Xu)
    assert np.allclose(J_onsager, f["J_n"], rtol=1e-8, atol=1e-16)


def test_paper1_limit(phys):
    """c_w = 0 reproduces Paper 1's solver_fd.py (short horizon, coarse grid)."""
    N = 40
    c_in = rh_to_conc(COND["rh_feed"], COND["T_in"])
    t_end = 0.3 * phys.stoichiometric_time(c_in, COND["T_in"])
    _, _, yp = generate_breakthrough_data(phys, N_z=N, t_final=t_end, c_in=c_in, n_snapshots=6,
                                          rtol=1e-9, atol=1e-13, verbose=False)
    col = Column(phys, N, c_w=0.0)
    _, ym = col.integrate(col.initial_state(phys.T_w), (0.0, t_end),
                          (c_in, phys.T_in, phys.T_w), 6, rtol=1e-9, atol=1e-13)
    n, a, u = col.split(ym)
    assert np.max(np.abs(n / col.eps - yp[:N])) < 1e-7 * c_in
    assert np.max(np.abs(a / col.A - yp[N:2 * N])) < 1e-6 * phys.q_max
    assert np.max(np.abs(col.temperature(n, a, u) - yp[2 * N:])) < 1e-5


@pytest.mark.parametrize("iso_name", ["paper1", "MOF-303", "Al-fumarate"])
def test_adsorbed_free_energy_derivatives(phys, iso_name):
    """mu_a = d f_a/dq, s_a = -d f_a/dT and e_a = q (e_w + dH), by finite differences."""
    col = Column(phys, 2, c_w=33.6, iso=iso_for(iso_name, phys))

    def f_a(q, T):
        x = col.X_of_q(np.array([q]))[0]
        phi = T * col.phi_over_T(T)
        return q * (phi + col.dH) + 8.314 * T * (q * np.log(x / (col.iso.b0 * 1.0)) - col.Pi_of_x(x))

    for q in (0.5, 3.0, 12.0, 20.0):
        for T in (295.0, 330.0, 360.0):
            hq, hT = 1e-5 * q, 1e-4
            mu_fd = (f_a(q + hq, T) - f_a(q - hq, T)) / (2 * hq)
            assert np.isclose(mu_fd / T, col.mu_a_over_T(np.array([q]), T)[0], rtol=1e-6)
            s_fd = -(f_a(q, T + hT) - f_a(q, T - hT)) / (2 * hT)
            assert np.isclose(s_fd, col.s_adsorbed(np.array([q]), T)[0], rtol=1e-6, atol=1e-6)
            e_a = f_a(q, T) + T * col.s_adsorbed(np.array([q]), T)[0]
            assert np.isclose(e_a, q * (col.e_w(T) + col.dH), rtol=1e-8)
