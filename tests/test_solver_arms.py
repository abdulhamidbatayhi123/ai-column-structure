"""Plan 2, Task 9: solver arms C (consistent hybrid scheme, coarse grids) and MUSCL-C."""
import numpy as np
import pytest

from p3b.generate import generate_trajectory
from p3b.simulate_a import setup_a
from p3b.solvers.coarse import MusclColumn, MusclStore, inject, run_c
from p3b.simulate_b import make_store

POINT_A = dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=0, n_cycles=1)
POINT_B = dict(T_hot=353.15, T_cold=298.15, v=2.0e-3, alpha=0.7, beta=0.8, index=0, n_cycles=1)
NS = 20


def test_inject_conserves_totals_and_splits_the_wall_heat():
    rng = np.random.default_rng(0)
    nvar, nc, r = 3, 4, 8
    y = rng.random((5, (nvar + 1) * nc))
    Y = inject(y, nvar, nc, r)
    assert Y.shape == (5, (nvar + 1) * nc * r)
    for j in range(nvar):
        np.testing.assert_allclose(Y[:, j * nc * r:(j + 1) * nc * r].sum(1) / r, y[:, j * nc:(j + 1) * nc].sum(1))
    np.testing.assert_allclose(Y[:, nvar * nc * r:].sum(1), y[:, nvar * nc:].sum(1))   # extensive: split


@pytest.mark.parametrize("system,point,material", [("A", POINT_A, "MOF-303"), ("B", POINT_B, None)])
def test_c_at_the_target_grid_is_the_coarse_reference(tmp_path, system, point, material):
    ref = tmp_path / "ref.npz"
    assert generate_trajectory(system, point, material, N_z=16, N_c=16, path=ref, n_snap=NS).ok
    res = run_c(system, point, material, N_c=16, path=tmp_path / "c.npz", n_snap=NS, n_target=16)
    assert res.ok
    a, b = np.load(ref), np.load(tmp_path / "c.npz")
    np.testing.assert_array_equal(a["channels"], b["channels"])
    np.testing.assert_array_equal(a["outlet"], b["outlet"])


def test_c_on_a_coarser_grid_writes_target_resolution(tmp_path):
    res = run_c("A", POINT_A, "MOF-303", N_c=8, path=tmp_path / "c.npz", n_snap=NS, n_target=16)
    assert res.ok
    d = np.load(tmp_path / "c.npz")
    assert d["channels"].shape == (1, NS, 16, 3)
    ch = d["channels"][0, 3]
    np.testing.assert_array_equal(ch[0::2], ch[1::2])          # piecewise constant: 2 target cells per C cell


def _faces(col, c):
    """Face concentrations the MUSCL column uses (faces 0..N), via its fluxes."""
    return col.face_values(c)


def test_muscl_face_values():
    col = MusclColumn.from_column(setup_a(POINT_A, "MOF-303", 16)[0])
    c_in = 2.0
    const = np.full(16, 1.0)
    np.testing.assert_allclose(col.face_values(const, c_in)[1:], 1.0)        # constant: upwind
    lin = np.linspace(1.0, 2.5, 16)
    f = col.face_values(lin, c_in)
    np.testing.assert_allclose(f[2:-1], 0.5 * (lin[1:-1] + lin[2:]))           # linear: exact face value
    peak = np.array([1.0, 2.0, 3.0, 1.0] * 4)
    f = col.face_values(peak, c_in)
    assert f[3] == peak[2]                                                    # extremum: limiter 0, upwind
    assert f[0] == c_in and f[-1] == peak[-1]                                 # inlet feed, outlet last cell
    assert col.consistent is False


def test_muscl_column_balances_hold():
    base, sched = setup_a(POINT_A, "MOF-303", 16)
    col = MusclColumn.from_column(base)
    z = (np.arange(16) + 0.5) / 16
    c = sched[0][2][0] * (0.1 + 0.8 / (1 + np.exp((z - 0.4) / 0.08)))
    T = 303.15 + 5 * z
    n, a = col.eps * c, col.A * col.q_star(c, T)
    b = col.budget(np.concatenate([n, a, col.energy(n, a, T)]), sched[0][2])
    assert abs(b["dN"] - b["portN"]) <= 1e-12 * (abs(b["dN"]) + abs(b["portN"]))
    assert abs(b["dE"] - b["portE"]) <= 1e-12 * (abs(b["dE"]) + abs(b["portE"]))


@pytest.mark.parametrize("v", [2.0e-3, -2.0e-3])
def test_muscl_store_conserves_energy(v):
    st = MusclStore.from_store(make_store(16))
    z = (np.arange(16) + 0.5) / 16
    Tf = 300 + 50 / (1 + np.exp((z - 0.5) / 0.1))
    y = np.concatenate([st.Cf * (Tf - 293.15), st.Cs * (Tf - 1 - 293.15)])
    b = st.budget(y, (v, 353.15 if v > 0 else 298.15, 293.15))
    assert abs(b["dE"] - b["portE"]) <= 1e-12 * (abs(b["dE"]) + abs(b["portE"]))
    assert st.consistent is False


def test_muscl_runs_a_cycle(tmp_path):
    res = run_c("A", POINT_A, "MOF-303", N_c=8, path=tmp_path / "m.npz", n_snap=NS, n_target=16, muscl=True)
    assert res.ok and np.load(tmp_path / "m.npz")["channels"].shape == (1, NS, 16, 3)
