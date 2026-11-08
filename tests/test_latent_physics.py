"""Plan 2, Task 1: physical functionals and ports of decoded states, against the numpy reference."""
import numpy as np
import pytest
import torch

from p3b.latent.physics import ColumnPhysics, StorePhysics
from p3b.simulate_a import setup_a
from p3b.simulate_b import T_AMB, make_store

N = 128
POINT = dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=0, n_cycles=1)


@pytest.fixture(scope="module", params=["MOF-303", "Al-fumarate"])
def case(request):
    col, schedule = setup_a(POINT, request.param, N)
    c_in = schedule[0][2][0]
    z = (np.arange(N) + 0.5) / N
    c = c_in * (0.05 + 0.9 / (1 + np.exp((z - 0.4) / 0.05)))
    T = 303.15 + 15.0 * np.exp(-((z - 0.5) / 0.2) ** 2)
    q = 0.8 * col.q_star(c, T)
    x = col.iso.X(q)
    n, a = col.eps * c, col.A * q
    u = col.energy(n, a, T)
    ch = torch.tensor(np.stack([np.log(c), np.log(x), np.log(T)], axis=-1), dtype=torch.float64)
    out = torch.tensor([np.log(c[-1]), np.log(col.temperature(n[-1:], a[-1:], u[-1:])[0])], dtype=torch.float64)
    return col, schedule, ch, out, (n, a, u)


def rel(a, b):
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-300)


def test_functionals_match_numpy_column(case):
    col, _, ch, _, (n, a, u) = case
    ph = ColumnPhysics(col)
    assert rel(ph.E(ch), np.sum(u) * col.dz) <= 1e-12
    assert rel(ph.N(ch), np.sum(n + a) * col.dz) <= 1e-12
    assert rel(ph.S(ch), np.sum(col.cell_entropy(n, a, u)) * col.dz) <= 1e-11


def test_ports_match_column_budget(case):
    col, schedule, ch, out, (n, a, u) = case
    ph = ColumnPhysics(col)
    for _, _, inputs in schedule:
        c_in, T_in, T_w = inputs
        b = col.budget(np.concatenate([n, a, u]), inputs)
        inp = torch.tensor([c_in, T_in, T_w, col.p.v], dtype=torch.float64)
        P_N, P_E, Phi_S = ph.ports(ch, out, inp)
        assert rel(P_N, b["portN"]) <= 1e-12
        assert rel(P_E, b["portE"]) <= 1e-12
        assert rel(Phi_S, b["portS"]) <= 1e-12


def test_batched_shapes(case):
    col, _, ch, out, _ = case
    ph = ColumnPhysics(col)
    chb, outb = ch.expand(5, -1, -1), out.expand(5, -1)
    inp = torch.tensor([[1.0, 303.15, 303.15, 0.1]] * 5, dtype=torch.float64)
    assert ph.E(chb).shape == (5,) and ph.S(chb).shape == (5,)
    assert all(p.shape == (5,) for p in ph.ports(chb, outb, inp))


def test_entropy_gradient_is_finite(case):
    col, _, ch, _, _ = case
    ph = ColumnPhysics(col)
    x = ch.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(ph.S(x), x)
    assert torch.isfinite(g).all() and g.abs().sum() > 0


@pytest.mark.parametrize("v", [2.0e-3, -2.0e-3])
def test_store_functionals_and_ports_match_numpy(v):
    st = make_store(N)
    z = (np.arange(N) + 0.5) / N
    Tf = 298.15 + 55.0 / (1 + np.exp((z - 0.3) / 0.04))
    Ts = Tf - 0.5 * np.sin(np.pi * z)
    y = np.concatenate([st.Cf * (Tf - 293.15), st.Cs * (Ts - 293.15)])
    T_in = 353.15 if v > 0 else 298.15
    b = st.budget(y, (v, T_in, T_AMB))
    ph = StorePhysics(st, T_AMB)
    ch = torch.tensor(np.stack([Tf, Ts], axis=-1), dtype=torch.float64)
    out = torch.tensor([Tf[-1] if v > 0 else Tf[0]], dtype=torch.float64)
    inp = torch.tensor([v, T_in], dtype=torch.float64)
    assert rel(ph.E(ch), np.sum(y) * st.dz) <= 1e-12
    S_np = np.sum(st.Cf * np.log(Tf / 293.15) + st.Cs * np.log(Ts / 293.15)) * st.dz
    assert rel(ph.S(ch), S_np) <= 1e-12
    P_E, Phi_S = ph.ports(ch, out, inp)
    assert rel(P_E, b["portE"]) <= 1e-12
    assert rel(Phi_S, b["portS"]) <= 1e-12
