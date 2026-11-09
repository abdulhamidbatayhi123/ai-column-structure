"""The reference simulators' exact Jacobian (Task 10 stall fix): the finite-difference Jacobian made
BDF stall on MOF-303 at the box centre (1024 cells: 642,420 evaluations; 2048 cells: no result in
11 h); with the exact Jacobian the same half-cycle took 45 CPU-s (verify/stall_exact_jacobian.py)."""
import numpy as np
import pytest
import torch
from scipy.optimize._numdiff import approx_derivative

from p3b.exact_jacobian import augmented_jacobian_a, augmented_jacobian_b
from p3b.simulate_a import _Augmented as AugA
from p3b.simulate_a import prewet_state, setup_a
from p3b.simulate_b import _Augmented as AugB
from p3b.simulate_b import make_store

POINT_A = dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=0, n_cycles=1)
N = 16


def _state_a():
    col, sch = setup_a(POINT_A, "MOF-303", N)
    z = (np.arange(N) + 0.5) / N
    c_in = sch[0][2][0]
    c = c_in * (0.05 + 0.9 / (1 + np.exp((z - 0.4) / 0.08)))
    T = 303.15 + 10 * np.exp(-((z - 0.5) / 0.2) ** 2)
    n, a = col.eps * c, col.A * 0.9 * col.q_star(c, T)
    y = np.concatenate([n, a, col.energy(n, a, T), np.zeros(N + 6)])
    return col, sch, y


def test_system_a_exact_jacobian_matches_finite_differences():
    col, sch, y = _state_a()
    aug = AugA(col)
    for _, _, inp in sch:
        J = augmented_jacobian_a(col, y, inp).toarray()
        Jfd = approx_derivative(lambda x: aug.rhs(0.0, x, inp), y, method="3-point")
        scale = np.abs(Jfd).max(axis=1, keepdims=True) + 1e-300
        assert np.all(np.abs(J - Jfd) <= 1e-5 * scale + 1e-12)
        assert J.shape == (4 * N + 6, 4 * N + 6) and np.all(J[:, 3 * N:] == 0)


@pytest.mark.parametrize("v", [2.0e-3, -2.0e-3])
def test_system_b_exact_jacobian_matches_finite_differences(v):
    st = make_store(N)
    z = (np.arange(N) + 0.5) / N
    Tf = 300 + 50 / (1 + np.exp((z - 0.5) / 0.1))
    y = np.concatenate([st.Cf * (Tf - 293.15), st.Cs * (Tf - 1 - 293.15), np.zeros(N + 4)])
    inp = (v, 353.15 if v > 0 else 298.15, 293.15)
    aug = AugB(st)
    J = augmented_jacobian_b(st, y, inp).toarray()
    Jfd = approx_derivative(lambda x: aug.rhs(0.0, x, inp), y, method="3-point")
    scale = np.abs(Jfd).max(axis=1, keepdims=True) + 1e-300
    assert np.all(np.abs(J - Jfd) <= 1e-5 * scale + 1e-12)


def test_simulators_use_the_exact_jacobian_and_close_the_balances():
    from p3b.simulate_a import simulate_trajectory_a
    res = simulate_trajectory_a(POINT_A, "MOF-303", 32, n_snap=20)
    assert res.ok and max(res.closure) < 1e-5
    from p3b import simulate_a
    assert simulate_a.EXACT_JACOBIAN is True
