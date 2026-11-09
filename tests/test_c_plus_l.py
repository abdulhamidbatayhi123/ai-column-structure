"""Plan 2, Task 10: C+L, the 32-cell consistent solver with learned non-negative multipliers."""
import numpy as np
import pytest
import torch

from p3b.simulate_a import prewet_state, setup_a
from p3b.simulate_b import T_AMB, make_store
from p3b.solvers.c_plus_l import CPlusL, Multipliers, TorchColumnRHS, TorchStoreRHS, implicit_euler_step

F64 = torch.float64
N = 32
POINT_A = dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=0, n_cycles=1)


def _state_a():
    col, sched = setup_a(POINT_A, "MOF-303", N)
    z = (np.arange(N) + 0.5) / N
    c_in = sched[0][2][0]
    c = c_in * (0.05 + 0.9 / (1 + np.exp((z - 0.4) / 0.06)))
    T = 303.15 + 12 * np.exp(-((z - 0.5) / 0.2) ** 2)
    n, a = col.eps * c, col.A * 0.9 * col.q_star(c, T)
    return col, sched, np.concatenate([n, a, col.energy(n, a, T)])


def test_torch_rhs_equals_the_numpy_column():
    col, sched, y = _state_a()
    rhs = TorchColumnRHS(col)
    for _, _, inp in sched:
        ref = col.rhs(0.0, y, inp)
        got = rhs(torch.tensor(y, dtype=F64), torch.tensor(inp, dtype=F64)).numpy()
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())


def test_unit_multipliers_give_c():
    col, sched, y = _state_a()
    rhs = TorchColumnRHS(col, Multipliers(n_out=3, constant=1.0))
    ref = col.rhs(0.0, y, sched[0][2])
    got = rhs(torch.tensor(y, dtype=F64), torch.tensor(sched[0][2], dtype=F64)).numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())


def test_multipliers_are_non_negative_for_any_weights():
    torch.manual_seed(0)
    m = Multipliers(n_out=3)
    for p in m.parameters():
        p.data.normal_(0, 5)
    out = m(torch.randn(1000, 3, dtype=F64))
    assert out.shape == (1000, 3) and torch.all(out >= 0)
    sizes = [l.weight.shape for l in m.nets[0].modules() if isinstance(l, torch.nn.Linear)]
    assert sizes == [(32, 3), (32, 32), (1, 32)]


def test_production_is_non_negative_with_learned_multipliers():
    torch.manual_seed(1)
    col, sched, y = _state_a()
    m = Multipliers(n_out=3)
    for p in m.parameters():
        p.data.normal_(0, 2)
    rhs = TorchColumnRHS(col, m)
    for _, _, inp in sched:
        sig = rhs.production(torch.tensor(y, dtype=F64), torch.tensor(inp, dtype=F64))
        assert all(float(v.min()) >= -1e-12 * float(v.abs().max() + 1e-300) for v in sig.values())


def test_implicit_euler_gradient_matches_finite_differences():
    torch.manual_seed(2)
    col, sched, y = _state_a()
    m = Multipliers(n_out=3)
    rhs = TorchColumnRHS(col, m)
    inp = torch.tensor(sched[0][2], dtype=F64)
    y0 = torch.tensor(y, dtype=F64)
    h = 0.5
    w = torch.randn(3 * N, dtype=F64)

    def loss():
        return (implicit_euler_step(rhs, y0, inp, h) * w).sum() / (y0.abs().max())

    p = next(m.parameters())
    L = loss()
    (g,) = torch.autograd.grad(L, p)
    e = torch.zeros_like(p)
    idx = (0,) * p.dim()
    e[idx] = 1e-6
    with torch.no_grad():
        p += e
        Lp = float(loss())
        p -= 2 * e
        Lm = float(loss())
        p += e
    assert float(g[idx]) == pytest.approx((Lp - Lm) / 2e-6, rel=1e-5, abs=1e-10)


def test_implicit_euler_solves_its_equation():
    col, sched, y = _state_a()
    rhs = TorchColumnRHS(col)
    inp = torch.tensor(sched[0][2], dtype=F64)
    y0 = torch.tensor(y, dtype=F64)
    y1 = implicit_euler_step(rhs, y0, inp, 2.0)
    res = y1 - y0 - 2.0 * rhs(y1, inp)
    assert float(res.abs().max()) <= 1e-10 * float(y0.abs().max())


@pytest.mark.parametrize("v", [2.0e-3, -2.0e-3])
def test_store_rhs_equals_numpy_and_unit_multipliers_give_c(v):
    st = make_store(N)
    z = (np.arange(N) + 0.5) / N
    Tf = 300 + 50 / (1 + np.exp((z - 0.5) / 0.1))
    y = np.concatenate([st.Cf * (Tf - 293.15), st.Cs * (Tf - 1 - 293.15)])
    inp = (v, 353.15 if v > 0 else 298.15, T_AMB)
    ref = st.rhs(0.0, y, inp)
    for rhs in (TorchStoreRHS(st), TorchStoreRHS(st, Multipliers(n_out=2, constant=1.0))):
        got = rhs(torch.tensor(y, dtype=F64), torch.tensor(inp, dtype=F64)).numpy()
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())


def test_c_plus_l_rollout_shapes():
    col, sched, _ = _state_a()
    model = CPlusL(TorchColumnRHS(col, Multipliers(n_out=3)))
    y0 = torch.tensor(prewet_state(col, sched[0][2][0], 303.15), dtype=F64)
    inps = torch.tensor([sched[0][2]] * 3, dtype=F64)
    Y = model.rollout(y0, inps, torch.full((3,), 1.0, dtype=F64))
    assert Y.shape == (3, 3 * N) and torch.isfinite(Y).all()
