"""Reference physics revision (plan 1, Task 7b; cs T3): flow-dependent axial heat dispersion and the
hybrid scheme (upwind plus physical diffusion lowered by the upwind truncation term, clipped at zero;
Spalding 1972, Patankar 1980) in both systems' reference solvers."""
import numpy as np
import pytest

from p3b import materials as m
from p3b.column import Column
from p3b.simulate_a import C_W
from p3b.thermal_store import ThermalStore

T_ADS = 303.15
C_IN = m.c_in_from_rh(0.40, T_ADS)


def column(N, v=0.10, compensate=True):
    iso = m.load_isotherm("MOF-303")
    phys = m.bed_physics(v, T_ADS)
    phys.k_LDF = m.k_ldf(iso, phys, C_IN, T_ADS)
    return Column(phys, N, C_W, iso=iso, compensate=compensate)


def random_state(col, rng):
    z = (np.arange(col.N) + 0.5) / col.N
    c = C_IN * (0.02 + 0.98 / (1 + np.exp((z - rng.uniform(0.2, 0.8)) * 30))) * (1 + 0.01 * rng.standard_normal(col.N))
    T = 300 + 25 * np.exp(-((z - rng.uniform(0.2, 0.8)) / 0.1) ** 2) + rng.normal(0, 0.5, col.N)
    q = 0.8 * col.q_star(c, T)
    n, a = c * col.eps, q * col.A
    return np.concatenate([n, a, col.energy(n, a, T)])


def test_bed_axial_conductivity_is_flow_dependent():
    for v in (0.03, 0.30):
        p = m.bed_physics(v, T_ADS)
        assert p.k_z == pytest.approx(m.K_E0 + m.DELTA_AX * p.rho_g * p.C_pg * v * p.d_p, rel=1e-15)
    assert m.K_E0 == 0.10 and m.DELTA_AX == 0.5


def test_hybrid_coefficients_per_face():
    rng = np.random.default_rng(0)
    for N in (16, 128):
        col = column(N, v=0.30)
        f = col.fluxes(random_state(col, rng), (C_IN, T_ADS, T_ADS))
        p, dz = col.p, col.dz
        assert f["epsD_eff"] == pytest.approx(max(0.0, col.eps * p.D_L - p.v * dz / 2), rel=1e-15)
        cbar = 0.5 * (f["c"][1:] + f["c"][:-1])
        np.testing.assert_allclose(f["k_eff"], np.maximum(0.0, p.k_z - (col.rhoC + cbar * C_W) * p.v * dz / 2), rtol=1e-15)
    assert np.all(column(16, v=0.30).fluxes(random_state(column(16), rng), (C_IN, T_ADS, T_ADS))["k_eff"] == 0.0)


def test_unclipped_hybrid_energy_convection_is_central():
    """Uniform c (no dispersion flux), unclipped faces: compensated upwind = central differencing, exactly."""
    col = column(512, v=0.10)
    N = col.N
    c = np.full(N, 0.5 * C_IN)
    z = (np.arange(N) + 0.5) / N
    T = 303.0 + 20.0 * np.exp(-((z - 0.5) / 0.1) ** 2)
    q = col.q_star(c, T)                       # equilibrium: no uptake
    n, a = col.eps * c, col.A * q
    y = np.concatenate([n, a, col.energy(n, a, T)])
    inputs = (0.5 * C_IN, T[0], 0.0)           # wall term removed below
    f = col.fluxes(y, inputs)
    assert np.all(f["k_eff"] > 0)
    du = col.split(col.rhs(0.0, y, inputs))[2] + f["W"]
    g = f["g_u"][1:]
    phys_cond = -(col.p.k_z) * np.diff(T) / col.dz
    div = np.zeros(N)
    div[:-1] -= phys_cond / col.dz
    div[1:] += phys_cond / col.dz
    central = col.p.v * (g[:-2] - g[2:]) / (2 * col.dz) + div[1:-1]
    np.testing.assert_allclose(du[1:-1], central, rtol=1e-9, atol=1e-9 * np.abs(central).max())


@pytest.mark.parametrize("N", [16, 40, 256])
def test_compensated_identities_and_production_signs(N):
    rng = np.random.default_rng(1)
    col = column(N, v=0.30)
    for inputs in [(C_IN, 303.15, 303.15), (C_IN, 353.15, 353.15)]:
        for _ in range(4):
            y = random_state(col, rng)
            b = col.budget(y, inputs)
            f = col.fluxes(y, inputs)
            dn, da, du = col.split(col.rhs(0.0, y, inputs))
            scale_S = np.sum(np.abs(col.mu_v_over_T(f["c"], f["T"]) * dn)
                             + np.abs(col.mu_a_over_T(f["q"], f["T"]) * da) + np.abs(du / f["T"])) * col.dz
            assert abs(b["dN"] - b["portN"]) <= 1e-12 * (abs(b["dN"]) + abs(b["portN"])) + 1e-14
            assert abs(b["dE"] - b["portE"]) <= 1e-11 * (abs(b["dE"]) + abs(b["portE"]))
            assert abs(b["dS"] - b["portS"] - b["sigma_total"]) <= 1e-10 * scale_S
            for key in ("sig_ldf", "sig_face", "sig_adv", "sig_wall"):
                assert np.min(b[key]) >= -1e-12 * (abs(b["sigma_total"]) / col.dz + 1e-30), key


def test_column_default_is_uncompensated():
    col = column(64, compensate=False)
    f = col.fluxes(random_state(col, np.random.default_rng(2)), (C_IN, T_ADS, T_ADS))
    assert f["epsD_eff"] == col.eps * col.p.D_L and np.all(f["k_eff"] == col.p.k_z)


# ---- system B ----------------------------------------------------------------------------

def store_state(st, rng):
    z = np.linspace(0, 1, st.N)
    Tf = 300 + 60 / (1 + np.exp((z - rng.uniform(0.2, 0.8)) * 25)) + rng.normal(0, 0.5, st.N)
    Ts = Tf - rng.normal(2.0, 1.0, st.N)
    return np.concatenate([st.Cf * (Tf - 293.15), st.Cs * (Ts - 293.15)])


def test_store_fluid_conductivity_is_flow_dependent_and_compensated():
    st = ThermalStore(1024, d_p=0.05, compensate=True)
    for v in (5e-4, -3e-3):
        k_phys = st.k_f + st.delta_ax * st.rhoc * abs(v) * 0.05
        assert st.k_f_phys(v) == pytest.approx(k_phys, rel=1e-15)
        assert st.k_f_eff(v) == pytest.approx(k_phys - st.rhoc * abs(v) * st.dz / 2, rel=1e-15)
    assert ThermalStore(64).k_f_eff(3e-3) == ThermalStore(64).k_f    # defaults unchanged


@pytest.mark.parametrize("v", [2e-3, -2e-3])
@pytest.mark.parametrize("N", [16, 256])
def test_store_compensated_identities_and_signs(v, N):
    rng = np.random.default_rng(3)
    st = ThermalStore(N, d_p=0.05, compensate=True)
    for _ in range(6):
        y = store_state(st, rng)
        b = st.budget(y, (v, 360.0 if v > 0 else 295.0, 293.15))
        assert abs(b["dE"] - b["portE"]) <= 1e-12 * (abs(b["dE"]) + abs(b["portE"]))
        assert abs(b["dS"] - b["portS"] - b["sigma"]) <= 1e-10 * b["scale"]
        for k, arr in b["sig"].items():
            assert np.min(arr) >= -1e-12 * (abs(b["sigma"]) / st.dz + 1e-30), k


def test_references_use_the_revised_physics():
    from p3b import simulate_a, simulate_b
    point = dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.30, alpha=0.8, beta=0.6, index=0, n_cycles=1)
    col, _ = simulate_a.setup_a(point, "MOF-303", 2048)
    assert col.compensate and col.p.k_z > m.K_E0
    f = col.fluxes(random_state(col, np.random.default_rng(4)), (C_IN, T_ADS, T_ADS))
    assert np.all(f["k_eff"] > 0) and f["epsD_eff"] > 0          # no clip at reference resolution
    st = simulate_b.make_store(1024)
    assert st.compensate and st.d_p == simulate_b.D_P_B and st.k_f_eff(3e-3) > 0
    assert st.k_f_eff(3e-3) < st.k_f_phys(3e-3)
