"""Coarse storage (plan 1, Task 6): block averages of conserved variables, channels, fine ports."""
import numpy as np
import pytest

from p3b import materials as m
from p3b.column import Column
from p3b.simulate_a import C_W
from p3b.store import Coarsener
from p3b.thermal_store import ThermalStore

N_Z, N_C, NS = 256, 32, 200
C_IN = m.c_in_from_rh(0.40, 303.15)


def column():
    iso = m.load_isotherm("MOF-303")
    phys = m.bed_physics(0.10, 303.15)
    phys.k_LDF = m.k_ldf(iso, phys, C_IN, 303.15)
    return Column(phys, N_Z, C_W, iso=iso)


def fine_states_a(col, seed=0, ns=NS):
    """Smooth breakthrough-like profiles (c, q below q*, T) and a cumulative wall heat per cell."""
    rng = np.random.default_rng(seed)
    z = (np.arange(N_Z) + 0.5) / N_Z
    Y = np.empty((ns, 4 * N_Z))
    for k in range(ns):
        front = 0.2 + 0.6 * k / ns + 0.01 * rng.standard_normal()
        c = C_IN * (1e-3 + 0.999 / (1 + np.exp((z - front) / 0.03)))
        T = 303.15 + 8.0 * np.exp(-((z - front) / 0.08) ** 2)
        q = 0.9 * col.q_star(c, T)
        n, a = col.eps * c, col.A * q
        Y[k] = np.concatenate([n, a, col.energy(n, a, T), 10.0 * k * np.ones(N_Z) / N_Z])
    return Y


def ports_a(ns=NS):
    return np.cumsum(np.ones((ns, 5)), axis=0)


def test_non_integer_factor_raises():
    with pytest.raises(ValueError):
        Coarsener("A", 2000, 128, model=column(), c_in=C_IN)


def test_totals_exact():
    col = column()
    cz = Coarsener("A", N_Z, N_C, model=col, c_in=C_IN)
    Y = fine_states_a(col)
    cz.add_cycle(0, np.arange(NS, dtype=float), Y, ports_a())
    blocks = cz.conserved[0]                                  # (NS, N_C, 3) float64
    dz_c = col.p.L / N_C
    for j in range(3):
        fine_tot = Y[:, j * N_Z:(j + 1) * N_Z].sum(axis=1) * col.dz
        np.testing.assert_allclose(blocks[:, :, j].sum(axis=1) * dz_c, fine_tot, rtol=1e-12)
    wall = cz.wall[0]
    np.testing.assert_allclose(wall.sum(axis=1), Y[:, 3 * N_Z:].sum(axis=1), rtol=1e-6)


def test_channels_finite_with_undershoot():
    col = column()
    cz = Coarsener("A", N_Z, N_C, model=col, c_in=C_IN)
    Y = fine_states_a(col, ns=2)
    Y[1, 5] = -1e-9 * col.eps                                 # one fine cell, one snapshot
    Y[0, :N_Z // N_C] = -1e-9 * col.eps                       # a whole block undershoots: floored channel
    cz.add_cycle(0, np.arange(2.0), Y, ports_a(2))
    assert np.all(np.isfinite(cz.channels[0]))
    assert cz.floored_cells == 1 + N_Z // N_C
    assert cz.checked_cells == 2 * N_Z


def test_no_fine_history():
    col = column()
    cz = Coarsener("A", N_Z, N_C, model=col, c_in=C_IN)
    for k in range(3):
        cz.add_cycle(k, np.arange(NS, dtype=float) + k * NS, fine_states_a(col, seed=k), ports_a())

    def arrays(obj):
        for v in vars(obj).values():
            if isinstance(v, np.ndarray):
                yield v
            elif isinstance(v, (list, tuple)):
                yield from (x for x in v if isinstance(x, np.ndarray))
    assert all(N_Z not in a.shape and 4 * N_Z not in a.shape for a in arrays(cz))
    assert all(a.dtype == np.float32 for a in cz.channels + cz.outlet)
    assert all(a.dtype == np.float64 for a in cz.wall)   # cumulative: float64 (cs T6 I7)


def test_roundtrip(tmp_path):
    col = column()
    cz = Coarsener("A", N_Z, N_C, model=col, c_in=C_IN)
    cz.add_cycle(0, np.arange(NS, dtype=float), fine_states_a(col), ports_a())
    path = tmp_path / "traj.npz"
    cz.to_npz(path, meta={"system": "A", "material": "MOF-303"})
    d = np.load(path, allow_pickle=False)
    assert d["channels"].shape == (1, NS, N_C, 3) and d["outlet"].shape == (1, NS, 2)
    assert d["wall_heat_blocks"].shape == (1, NS, N_C) and d["ports_cum"].shape == (1, NS, 5)
    assert d["conserved_blocks"].dtype == np.float64 and d["t"].shape == (1, NS)
    assert "commit" in str(d["meta"])
    ch = d["channels"][0].astype(float)
    c, x, T = np.exp(ch[..., 0]), np.exp(ch[..., 1]), np.exp(ch[..., 2])
    n, a = col.eps * c, col.A * col.iso.Q(x)
    u = col.energy(n, a, T)
    ref = d["conserved_blocks"][0]
    for j, val in enumerate((n, a, u)):
        np.testing.assert_allclose(val, ref[..., j], rtol=0, atol=1e-5 * np.abs(ref[..., j]).max())


def test_system_b_channels_and_outlet():
    st = ThermalStore(N_Z)
    cz = Coarsener("B", N_Z, N_C, model=st)
    Tf = np.linspace(350.0, 300.0, N_Z)
    y = np.concatenate([st.Cf * (Tf - 293.15), st.Cs * (Tf - 1.0 - 293.15), np.zeros(N_Z)])
    Y = np.tile(y, (4, 1))
    ports = np.zeros((4, 4))
    ports[:, 3] = [1, 1, -1, -1]
    cz.add_cycle(0, np.arange(4.0), Y, ports)
    assert cz.channels[0].shape == (4, N_C, 2)
    np.testing.assert_allclose(cz.outlet[0][:, 0], [Tf[-1], Tf[-1], Tf[0], Tf[0]], rtol=1e-6)
    # both fine ends, float64, so a gate can use the outlet of the interval's own flow direction
    assert cz.ends[0].dtype == np.float64 and cz.ends[0].shape == (4, 2)
    np.testing.assert_allclose(cz.ends[0], np.tile([Tf[0], Tf[-1]], (4, 1)), rtol=1e-14)
