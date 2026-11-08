"""System A cycle simulator (plan 1, Task 4). N_z = 64 and 2 cycles so the tests run in seconds."""
import numpy as np
import pytest

from p3b import materials as m
from p3b import simulate_a as sa

POINT = dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=0, n_cycles=2)
N_Z = 64


@pytest.fixture(scope="module")
def run():
    calls = []
    res = sa.simulate_trajectory_a(POINT, "MOF-303", N_Z, on_cycle=lambda *a: calls.append(a))
    return res, calls


def test_prewet_initial_state(run):
    res, calls = run
    assert res.ok, res.message
    k, t, y, ports = calls[0]
    assert k == 0 and t[0] == 0.0 and y.shape == (200, 4 * N_Z) and ports.shape == (200, sa.N_PORTS)
    c_in = m.c_in_from_rh(0.40, 303.15)
    iso = m.load_isotherm("MOF-303")
    phys = m.bed_physics(0.10, 303.15)
    eps, A = phys.eps_t, (1 - phys.eps_t) * phys.rho_p
    n, a = y[0, :N_Z], y[0, N_Z:2 * N_Z]
    np.testing.assert_allclose(n / eps, 1e-4 * c_in, rtol=1e-12)
    q0 = float(iso.q_star(np.array([1e-4 * c_in]), np.array([303.15]))[0])
    np.testing.assert_allclose(a / A, q0, rtol=1e-12)
    assert np.all(ports[0] == 0.0)


def test_desorption_feed_same_absolute_humidity():
    iso = m.load_isotherm("MOF-303")
    phys = m.bed_physics(POINT["v"], POINT["T_ads"])
    c_in = m.c_in_from_rh(POINT["RH_ads"], POINT["T_ads"])
    (ph1, d1, in1), (ph2, d2, in2) = sa.cycle_schedule(POINT, iso, phys)
    assert (ph1, ph2) == ("ads", "des")
    assert in1 == (c_in, 303.15, 303.15)
    assert in2 == (c_in, 353.15, 353.15)
    t_st = m.t_stoich(iso, phys, c_in, 303.15)
    assert d1 == pytest.approx(0.8 * t_st) and d2 == pytest.approx(0.6 * 0.8 * t_st)


def test_integrated_closure(run):
    res, _ = run
    assert len(res.closure) == 2 * POINT["n_cycles"]
    assert max(res.closure) <= 1e-5, res.closure


def test_cycle_times_are_contiguous(run):
    _, calls = run
    t0, t1 = calls[0][1], calls[1][1]
    assert np.all(np.diff(t0) > 0) and t1[0] > t0[-1]
    np.testing.assert_allclose(calls[1][3][0, :2], calls[0][3][-1, :2], rtol=0.05)  # ports cumulative


def test_wall_heat_per_cell_sums_to_wall_port(run):
    _, calls = run
    for _, _, y, ports in calls:
        np.testing.assert_allclose(y[:, 3 * N_Z:].sum(axis=1), ports[:, 4], rtol=1e-12, atol=1e-9)


def test_jacobian_pattern_is_banded():
    col_states = 4 * N_Z + 4
    S = sa.augmented_sparsity(N_Z)
    assert S.shape == (col_states, col_states)
    assert S.getnnz(axis=1).max() <= 12   # no dense row: colored finite differences stay cheap


def test_failure_is_recorded():
    res = sa.simulate_trajectory_a(dict(POINT, v=0.0), "MOF-303", N_Z)
    assert res.ok is False and res.failed_cycle is not None and res.message


def test_callback_once_per_cycle(run):
    _, calls = run
    assert [c[0] for c in calls] == list(range(POINT["n_cycles"]))
