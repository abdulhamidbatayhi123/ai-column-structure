"""Property tests of System B, the two-temperature thermocline store (verify/thermal_store.py)."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "verify"))
from thermal_store import ThermalStore  # noqa: E402


def random_state(st, rng):
    z = np.linspace(0, 1, st.N)
    Tf = 300 + 60 / (1 + np.exp((z - rng.uniform(0.2, 0.8)) * 25)) + rng.normal(0, 0.5, st.N)
    Ts = Tf - rng.normal(2.0, 1.0, st.N)
    return np.concatenate([st.Cf * (Tf - 293.15), st.Cs * (Ts - 293.15)])


@pytest.mark.parametrize("v", [2e-3, -2e-3])
def test_identities_and_signs_both_directions(v):
    rng = np.random.default_rng(3)
    st = ThermalStore(60)
    for _ in range(10):
        y = random_state(st, rng)
        b = st.budget(y, (v, 360.0 if v > 0 else 295.0, 293.15))
        assert abs(b["dE"] - b["portE"]) <= 1e-12 * (abs(b["dE"]) + abs(b["portE"]))
        assert abs(b["dS"] - b["portS"] - b["sigma"]) <= 1e-10 * b["scale"]
        for k, arr in b["sig"].items():
            assert np.min(arr) >= -1e-12 * (abs(b["sigma"]) / st.dz + 1e-30), k


def test_charge_discharge_cycle_closes():
    """Integrated energy closure over a charge (from z=0) and a discharge (from z=L) step."""
    st = ThermalStore(80)
    y = st.initial_state(295.0)
    t_front = st.L * (st.Cf + st.Cs) / (st.rhoc * 2e-3)     # thermal front transit time
    for inputs in ((2e-3, 360.0, 293.15), (-2e-3, 295.0, 293.15)):
        ts, ys = st.integrate(y, (0.0, 0.7 * t_front), inputs, 401)
        E = np.array([np.sum(ys[:, k]) * st.dz for k in range(ys.shape[1])])
        portE = np.array([st.budget(ys[:, k], inputs)["portE"] for k in range(ys.shape[1])])
        gained = np.trapezoid(portE, ts)
        assert abs(E[-1] - E[0] - gained) <= 2e-4 * abs(gained)
        y = ys[:, -1]
