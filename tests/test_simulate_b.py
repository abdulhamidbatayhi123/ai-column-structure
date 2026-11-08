"""System B cycle simulator (plan 1, Task 5): thermocline store with flow reversal. N_z = 64."""
import numpy as np
import pytest

from p3b import simulate_b as sb
from p3b.thermal_store import ThermalStore

POINT = dict(T_hot=353.15, T_cold=298.15, v=2.0e-3, alpha=0.7, beta=0.8, index=0, n_cycles=2)
N_Z = 64


@pytest.fixture(scope="module")
def run():
    calls = []
    res = sb.simulate_trajectory_b(POINT, N_Z, on_cycle=lambda *a: calls.append(a))
    return res, calls


def test_initial_uniform(run):
    res, calls = run
    assert res.ok, res.message
    k, t, y, ports = calls[0]
    assert k == 0 and t[0] == 0.0 and y.shape == (200, 3 * N_Z) and ports.shape == (200, sb.N_PORTS)
    st = ThermalStore(N_Z)
    Tf, Ts = st.temps(y[0, :2 * N_Z])
    np.testing.assert_allclose(Tf, 298.15, rtol=1e-14)
    np.testing.assert_allclose(Ts, 298.15, rtol=1e-14)


def test_flow_reverses(run):
    st = ThermalStore(N_Z)
    (ph1, d1, in1), (ph2, d2, in2) = sb.cycle_schedule(POINT, st)
    assert (ph1, ph2) == ("charge", "discharge")
    assert in1 == (2.0e-3, 353.15, sb.T_AMB) and in2 == (-2.0e-3, 298.15, sb.T_AMB)
    t_front = st.L * (st.Cf + st.Cs) / (st.rhoc * 2.0e-3)
    assert d1 == pytest.approx(0.7 * t_front) and d2 == pytest.approx(0.8 * 0.7 * t_front)
    _, calls = run
    direction = calls[0][3][:, 3]
    assert np.all(direction[:100] == 1.0) and np.all(direction[100:] == -1.0)


def test_charge_stores_heat_at_inlet_end(run):
    _, calls = run
    st = ThermalStore(N_Z)
    Tf, _ = st.temps(calls[0][2][99, :2 * N_Z])   # end of the first charge
    assert Tf[0] > Tf[-1] + 10.0


def test_integrated_closure(run):
    res, _ = run
    assert len(res.closure) == 2 * POINT["n_cycles"]
    assert max(res.closure) <= 1e-5, res.closure


def test_wall_heat_per_cell_sums_to_wall_port(run):
    _, calls = run
    for _, _, y, ports in calls:
        np.testing.assert_allclose(y[:, 2 * N_Z:].sum(axis=1), ports[:, 2], rtol=1e-12, atol=1e-6)


def test_failure_is_recorded():
    res = sb.simulate_trajectory_b(dict(POINT, v=0.0), N_Z)
    assert res.ok is False and res.failed_cycle is not None and res.message


def test_callback_once_per_cycle(run):
    _, calls = run
    assert [c[0] for c in calls] == list(range(POINT["n_cycles"]))
