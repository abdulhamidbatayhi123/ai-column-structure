"""Revision R3 (author's decision 2026-11-12: the physically right delta and h_v): one consistent
packed-bed set for both systems (Vortmeyer 1989).

  * fluid-to-particle coefficient: Wakao, Kaguei and Funazkri (1979), Nu = 2 + 1.1 Pr^(1/3) Re^0.6,
    with the fluid dispersion 0.5 Pr Re k_f it was derived with (Wakao and Kaguei 1982);
  * intraparticle conduction: Jeffreson (1972), 1/h_eff = 1/h + d_p/(10 k_s);
  * system B (two temperatures): h_v(|v|) = a_s h_eff, a_s = 6 (1 - eps)/d_p;
  * system A (one temperature): the Vortmeyer and Schaefer (1974) equivalent of the same set,
    k_z = k_e0 + 0.5 rho_g C_pg v d_p + Phi (rho_g C_pg v)^2/(h_eff a_s), Phi = (kappa/(1 + kappa))^2.

The fluid properties are taken at the state each model's constant density and heat capacity describe
(verify/r3/properties.py: air at 293.15 K, water at 318.15 K)."""
import json
import math
import os

import pytest

from p3b import gates
from p3b import materials as m
from p3b import packed_bed as pb
from p3b import simulate_b as sb

ROOT = os.path.join(os.path.dirname(__file__), "..")
V_A = (0.03, 0.05, 0.10, 0.20, 0.30)                  # system A design box, m/s (superficial)
V_B = (0.5e-3, 1.0e-3, 2.0e-3, 3.0e-3)                # system B design box


def _nu(re, pr):
    return 2.0 + 1.1 * pr ** (1.0 / 3.0) * re ** 0.6


def _h(speed, d_p, rho, c, mu, k, k_s):
    h = _nu(rho * abs(speed) * d_p / mu, c * mu / k) * k / d_p
    return h if k_s is None else 1.0 / (1.0 / h + d_p / (10.0 * k_s))


def test_the_fluid_properties_are_the_recorded_computation():
    rec = json.load(open(os.path.join(ROOT, "verify", "r3", "properties.json")))["states"]
    assert pb.AIR_A == {"mu": rec["air_A"]["mu_Pa_s"], "k": rec["air_A"]["k_W_mK"]}
    assert pb.WATER_B == {"mu": rec["water_B"]["mu_Pa_s"], "k": rec["water_B"]["k_W_mK"]}
    assert rec["air_A"]["T_K"] == 293.15 and rec["water_B"]["T_K"] == 318.15
    # the states are the ones the models' constant density and heat capacity describe
    assert rec["water_B"]["rho_kg_m3"] == pytest.approx(990.0, rel=1e-3)
    assert rec["water_B"]["cp_J_kgK"] == pytest.approx(4180.0, rel=1e-3)
    assert rec["air_A"]["rho_kg_m3"] == pytest.approx(1.2, rel=5e-3)


def test_wakao_nusselt_and_its_stagnant_limit():
    for re, pr in ((0.0, 0.7), (15.0, 0.7), (250.0, 3.9)):
        assert pb.nusselt_wakao(re, pr) == pytest.approx(_nu(re, pr), rel=1e-15)
    assert pb.nusselt_wakao(0.0, 0.7) == 2.0


def test_jeffreson_correction_and_the_specific_area():
    assert pb.jeffreson(300.0, 0.05, 2.5) == pytest.approx(1.0 / (1.0 / 300.0 + 0.05 / 25.0), rel=1e-15)
    assert pb.jeffreson(300.0, 0.05, None) == 300.0
    assert pb.jeffreson(300.0, 0.05, 2.5) < 300.0
    assert pb.specific_area(0.38, 0.05) == pytest.approx(6 * 0.62 / 0.05, rel=1e-15)


def test_system_a_kz_is_the_vortmeyer_schaefer_equivalent_of_the_two_phase_set():
    for v in V_A:
        p = m.bed_physics(v, 303.15)
        rho_c = p.rho_g * p.C_pg
        kappa = (1 - p.eps_t) * p.rho_p * p.C_ps / (p.eps_t * rho_c)
        h = _h(v, p.d_p, p.rho_g, p.C_pg, pb.AIR_A["mu"], pb.AIR_A["k"], pb.K_PELLET)
        a_s = 6 * (1 - p.eps_t) / p.d_p
        want = m.K_E0 + 0.5 * rho_c * v * p.d_p + (kappa / (1 + kappa)) ** 2 * (rho_c * v) ** 2 / (h * a_s)
        assert p.k_z == pytest.approx(want, rel=1e-12)
        assert m.DELTA_AX == 0.5                      # the fluid-phase dispersion (Wakao and Kaguei)


def test_system_a_delta_eff_starts_at_yagis_measured_level_and_rises_with_flow():
    def delta_eff(v):
        p = m.bed_physics(v, 303.15)
        return (p.k_z - m.K_E0) / (p.rho_g * p.C_pg * v * p.d_p)
    d = [delta_eff(v) for v in V_A]
    assert 0.70 <= d[0] <= 0.85                       # Yagi et al. 1960: 0.7 to 0.8 (one-phase, measured)
    assert all(b > a for a, b in zip(d, d[1:]))
    assert 1.2 <= d[-1] <= 1.9


def test_system_a_kz_uses_reference_state_properties_at_every_phase_temperature():
    assert m.bed_physics(0.1, 293.15).k_z == m.bed_physics(0.1, 363.15).k_z


def test_system_a_kz_at_rest_is_the_stagnant_conductivity():
    p = m.bed_physics(0.1, 303.15)
    assert pb.one_temperature_kz(0.0, m.K_E0, p.eps_t, p.d_p, p.rho_g, p.C_pg, pb.AIR_A["mu"], pb.AIR_A["k"],
                                 (1 - p.eps_t) * p.rho_p * p.C_ps, pb.K_PELLET, m.DELTA_AX) == m.K_E0


def test_system_b_exchange_follows_wakao_and_jeffreson():
    st = sb.make_store(16)
    a_s = 6 * (1 - st.params["eps"]) / sb.D_P_B
    hv = []
    for v in V_B:
        want = a_s * _h(v, sb.D_P_B, st.params["rho_f"], st.params["c_f"], pb.WATER_B["mu"], pb.WATER_B["k"],
                        pb.K_ROCK)
        assert st.h_v_of(v) == pytest.approx(want, rel=1e-12)
        assert st.h_v_of(-v) == st.h_v_of(v)          # the flow reverses between charge and discharge
        hv.append(st.h_v_of(v))
    assert all(b > a for a, b in zip(hv, hv[1:]))
    assert 1.0e4 <= hv[0] and hv[-1] <= 2.6e4         # bracket around the old constant 2e4 (T9, Section 4)
    assert st.h_v_of(0.0) > 0 and math.isfinite(st.h_v_of(0.0))


def test_the_exchange_model_uses_the_stores_own_constants():
    st = sb.make_store(16)
    spec = st.params["h_v_model"]
    assert spec["eps"] == st.params["eps"] and spec["rho"] == st.params["rho_f"] and spec["c"] == st.params["c_f"]
    assert spec["d_p"] == st.params["d_p"] == sb.D_P_B and spec["k_s"] == pb.K_ROCK


def test_the_constants_record_both_models_with_their_parameters():
    a = json.loads(json.dumps(gates.current_constants("A")))
    assert a["K_Z_MODEL"] == json.loads(json.dumps(pb.kz_spec()))
    assert set(a["K_Z_MODEL"]) >= {"name", "mu", "k", "k_s", "delta_f"}
    b = json.loads(json.dumps(gates.current_constants("B")))
    spec = b["store"]["h_v_model"]
    assert spec["name"] and set(spec) >= {"d_p", "eps", "rho", "c", "mu", "k", "k_s"}
    assert all(math.isfinite(x) for x in (spec["mu"], spec["k"], spec["k_s"]))
