"""Fixes from the whole-branch review of Plan 2 (one test per finding), and the PREREG follow-ups
(cs-org1 review T5: F3 S-L port term, F2 (i) conservative reconstruction for C)."""
import numpy as np
import pytest
import torch

from p3b.latent.arms_structured import SDynamics, SLDynamics, projected_rk4_step, structure_terms, structured_step
from p3b.latent.arms_unstructured import UDynamics, UFDynamics, count_params, match_width
from p3b.latent.autoencoder import MLPAutoencoder
from p3b.latent.data import Standardiser
from p3b.latent.fno import FNO1d
from p3b.latent.functionals import LatentFunctionals
from p3b.latent.integrate import balance_project
from p3b.latent.physics import ColumnPhysics
from p3b.simulate_a import setup_a
from p3b.solvers.c_plus_l import Multipliers, TorchColumnRHS, colored_jacobian, implicit_euler_step
from p3b.solvers.coarse import reconstruct_minmod

F64 = torch.float64
POINT_A = dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=0, n_cycles=1)


def _lf(nc=8, d=6, seed=4):
    torch.manual_seed(seed)
    col, sched = setup_a(POINT_A, "MOF-303", nc)
    n = nc * 3 + 2
    mean = torch.tensor([0.0, -1.0, 5.74] * nc + [0.0, 5.74], dtype=F64)
    std = Standardiser(mean, torch.full((n,), 0.05, dtype=F64))
    lf = LatentFunctionals(MLPAutoencoder(n, d).decode, std, ColumnPhysics(col), "A", nc)
    c_in, T_in, T_w = sched[0][2]
    return lf, torch.tensor([[c_in, T_in, T_w, 0.1]] * 3, dtype=F64)


# ---- Critical 1: the rank test and the projector must not depend on units -------------------
def test_rank_test_is_unit_free():
    torch.manual_seed(0)
    a, b = torch.randn(3, 6, 1, dtype=F64), torch.randn(3, 6, 1, dtype=F64)
    B = torch.cat([1e9 * a, b], -1)                      # unscaled singular-value ratio ~1e-9 < 1e-8
    t = torch.randn(3, 2, dtype=F64) * torch.tensor([1e9, 1.0], dtype=F64)
    out, info = structure_terms(B, torch.randn(3, 6, dtype=F64), torch.randn(3, 6, dtype=F64),
                                torch.randn(3, 6, 6, dtype=F64), t, torch.randn(3, dtype=F64), kappa=1e-6)
    assert not info["rank_deficient"].any()
    assert torch.allclose((B * out.unsqueeze(-1)).sum(-2), t, rtol=1e-9)


def test_real_column_b_is_well_conditioned_after_scaling():
    lf, raw = _lf(nc=128, d=32, seed=1)
    z = 0.3 * torch.randn(3, 32, dtype=F64)
    t = lf.evaluate(z, raw, grads=True)
    B = torch.stack([t["dE"], t["dN"]], -1)
    Bs = B / B.norm(dim=-2, keepdim=True)
    s = torch.linalg.svdvals(Bs)
    assert float((s[:, -1] / s[:, 0]).min()) > 1e-4
    _, info = structure_terms(B, t["dS"], torch.zeros_like(z), None, torch.stack([t["P_E"], t["P_N"]], -1),
                              t["Phi_S"], kappa=1e-6, metriplectic=False)
    assert not info["rank_deficient"].any()


def test_balance_project_is_unit_free():
    a = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=F64)
    b = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=F64)
    cons = lambda x: torch.stack([1e10 * (x @ a), x @ b], -1)  # noqa: E731
    z = torch.zeros(2, 4, dtype=F64)
    target = torch.tensor([[1e10, 1.0], [2e10, -1.0]], dtype=F64)
    zp, info = balance_project(z, cons, target)
    assert not info["rank_deficient"].any() and info["converged"].all()
    assert torch.allclose(cons(zp), target, rtol=1e-12)


# ---- Important 9: per-sample convergence with a physical scale -------------------------------
def test_balance_project_flags_per_sample_convergence():
    cons = lambda x: (x ** 2).sum(-1, keepdim=True)  # noqa: E731
    z = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=F64)        # second sample: zero Jacobian
    target = torch.tensor([[1.0], [1.0]], dtype=F64)
    _, info = balance_project(z, cons, target, iters=5)
    assert info["converged"].tolist() == [True, False]


def test_balance_project_uses_the_given_scale():
    cons = lambda x: x.sum(-1, keepdim=True)  # noqa: E731
    z = torch.zeros(1, 3, dtype=F64)
    _, info = balance_project(z, cons, torch.zeros(1, 1, dtype=F64), scale=torch.ones(1, 1, dtype=F64))
    assert info["converged"].all()                                   # target 0: relative test impossible


# ---- Important 2: the projected step is differentiable during training -----------------------
def test_projection_gradient_is_exact_for_linear_constraints():
    torch.manual_seed(8)
    a, b = torch.randn(5, dtype=F64), torch.randn(5, dtype=F64)
    cons = lambda x: torch.stack([x @ a, x @ b], -1)  # noqa: E731
    W = torch.randn(5, 5, dtype=F64, requires_grad=True)
    z0, tgt = torch.randn(2, 5, dtype=F64), torch.randn(2, 2, dtype=F64)
    w = torch.randn(2, 5, dtype=F64)

    def loss(Wm):
        zp, _ = balance_project(z0 @ Wm, cons, tgt)
        return (zp * w).sum()

    (g,) = torch.autograd.grad(loss(W), W)
    e = torch.zeros_like(W)
    e[1, 2] = 1e-6
    fd = (float(loss(W.detach() + e)) - float(loss(W.detach() - e))) / 2e-6
    assert float(g[1, 2]) == pytest.approx(fd, rel=1e-7)


def test_s_training_gradient_through_projected_steps_matches_finite_differences():
    """S's vector field meets both balances in continuous time, so an RK4 step leaves the constraint
    set by O(h^5) and the first-order projection gradient is accurate. (An arm whose field ignores
    the balances, like S-Pi, gets a gradient accurate to first order in the distance projected.)"""
    lf, raw = _lf()
    torch.manual_seed(9)
    s = SDynamics(6, 4, lf, kappa=1e-10)
    torch.nn.init.normal_(s.G[-1].weight, std=0.1)         # G starts at zero (Plan 3 init); give it a slope
    for p_ in s.parameters():
        p_.data.mul_(0.1)
    feat = torch.randn(3, 4, dtype=F64)
    z0 = 0.1 * torch.randn(3, 6, dtype=F64)
    w = torch.randn(3, 6, dtype=F64)

    def loss():
        z = z0
        for _ in range(3):
            z, _ = structured_step(s, z, feat, raw, 0.5)
        return (z * w).sum()

    p = s.G[0].weight
    (g,) = torch.autograd.grad(loss(), p)
    assert float(g.abs().max()) > 0
    e = torch.zeros_like(p)
    e[0, 0] = 1e-2          # large enough that the projection's 1e-12 convergence noise does not dominate
    with torch.no_grad():
        p += e
        Lp = float(loss())
        p -= 2 * e
        Lm = float(loss())
        p += e
    assert float(g[0, 0]) == pytest.approx((Lp - Lm) / 2e-2, rel=5e-3)


# ---- Important 3: functional values keep their dependence on z in training --------------------
def test_functionals_stay_attached_when_grad_is_enabled():
    lf, raw = _lf()
    z = 0.1 * torch.randn(3, 6, dtype=F64, requires_grad=True)
    t = lf.evaluate(z, raw)
    assert t["S"].requires_grad and t["dS"].requires_grad
    uf = UFDynamics(6, 4, 16, "A")
    torch.nn.init.normal_(uf.net[-1].weight, std=0.1)       # starts at zero (Plan 3 init); give it a slope
    out = uf(z, torch.zeros(3, 4, dtype=F64), {k: v / (v.detach().abs().mean() + 1) for k, v in t.items()})
    (gz,) = torch.autograd.grad(out.sum(), z)
    t_det = {k: v.detach() / (v.detach().abs().mean() + 1) for k, v in t.items()}
    (gz_det,) = torch.autograd.grad(uf(z, torch.zeros(3, 4, dtype=F64), t_det).sum(), z)
    assert not torch.allclose(gz, gz_det)


def test_decoder_weights_are_not_trained_through_the_functionals():
    lf, raw = _lf()
    z = 0.1 * torch.randn(3, 6, dtype=F64, requires_grad=True)
    lf.evaluate(z, raw)["S"].sum().backward()
    assert all(p.grad is None for p in lf.decode.__self__.parameters())


# ---- Important 4 and 5: C+L gradient test that can fail; Newton never fails silently -----------
def _state_a(N=32):
    col, sched = setup_a(POINT_A, "MOF-303", N)
    z = (np.arange(N) + 0.5) / N
    c_in = sched[0][2][0]
    c = c_in * (0.05 + 0.9 / (1 + np.exp((z - 0.4) / 0.06)))
    T = 303.15 + 12 * np.exp(-((z - 0.5) / 0.2) ** 2)
    n, a = col.eps * c, col.A * 0.9 * col.q_star(c, T)
    return col, sched, torch.tensor(np.concatenate([n, a, col.energy(n, a, T)]), dtype=F64)


def test_c_plus_l_gradient_is_nonzero_and_matches_finite_differences():
    torch.manual_seed(5)
    col, sched, y0 = _state_a()
    m = Multipliers(n_out=3)
    for net in m.nets:
        net[-1].weight.data.normal_(0, 1.0)
    rhs = TorchColumnRHS(col, m)
    inp = torch.tensor(sched[0][2], dtype=F64)
    w = torch.randn(y0.numel(), dtype=F64) / y0.abs().clamp(min=1e-3)

    def loss():
        return (implicit_euler_step(rhs, y0, inp, 5.0) * w).sum()

    p = m.nets[2][0].bias
    (g,) = torch.autograd.grad(loss(), p)
    assert float(g.abs().max()) > 0
    e = torch.zeros_like(p)
    e[0] = 1e-5
    with torch.no_grad():
        p += e
        Lp = float(loss())
        p -= 2 * e
        Lm = float(loss())
        p += e
    assert float(g[0]) == pytest.approx((Lp - Lm) / 2e-5, rel=1e-4)


def test_newton_failure_raises():
    col, sched, y0 = _state_a()
    rhs = TorchColumnRHS(col)
    with pytest.raises(RuntimeError, match="Newton"):
        implicit_euler_step(rhs, y0, torch.tensor(sched[0][2], dtype=F64), 50.0, max_iter=1)


# ---- Important 6: a cheap Jacobian and substeps ---------------------------------------------
def test_colored_jacobian_equals_the_dense_jacobian():
    torch.manual_seed(6)
    col, sched, y0 = _state_a()
    m = Multipliers(n_out=3)
    for net in m.nets:
        net[-1].weight.data.normal_(0, 1.0)
    rhs = TorchColumnRHS(col, m)
    inp = torch.tensor(sched[0][2], dtype=F64)
    dense = torch.autograd.functional.jacobian(lambda x: rhs(x, inp), y0)
    J = colored_jacobian(lambda x: rhs(x, inp), y0, n_cells=32, n_var=3)
    assert torch.allclose(J, dense, rtol=1e-12, atol=1e-12 * float(dense.abs().max()))


def test_substeps_converge_to_the_continuous_solution_at_first_order():
    col, sched, y0 = _state_a()
    rhs = TorchColumnRHS(col)
    inp = torch.tensor(sched[0][2], dtype=F64)
    from scipy.integrate import solve_ivp
    ref = solve_ivp(lambda t, y: col.rhs(t, y, sched[0][2]), (0, 20.0), y0.numpy(), method="BDF",
                    rtol=1e-10, atol=1e-12).y[:, -1]
    errs = []
    for k in (4, 8):
        y = implicit_euler_step(rhs, y0, inp, 20.0, substeps=k)
        errs.append(float(np.abs(y.detach().numpy() - ref).max()))
    assert 1.6 < errs[0] / errs[1] < 2.4


# ---- Important 7: the matcher never builds full-size modules ---------------------------------
def test_match_width_builds_on_the_meta_device():
    seen = []

    def build(w):
        seen.append(torch.empty(1).device.type)
        return FNO1d(128, 3, 2, 4, width=w)

    target = count_params(FNO1d(128, 3, 2, 4, width=20))
    w = match_width(build, target)
    assert set(seen) == {"meta"} and abs(w - 20) <= 1


# ---- Important 8: structured steps keep the defect counts of every stage ----------------------
def test_structured_step_returns_the_stage_flags():
    lf, raw = _lf()
    s = SDynamics(6, 4, lf, kappa=1e-10)
    z = 0.1 * torch.randn(3, 6, dtype=F64)
    z1, info = structured_step(s, z, torch.randn(3, 4, dtype=F64), raw, 2.0)
    assert z1.shape == z.shape
    for key in ("rank_deficient", "kappa_binds", "residual_deficit", "projection_rank_deficient",
                "projection_converged"):
        assert info[key].shape == (3,)


# ---- PREREG F3: S-L's port term is S's own projection --------------------------------------
def test_sl_port_term_is_s_projection():
    lf, raw = _lf()
    torch.manual_seed(7)
    sl = SLDynamics(6, 4, lf, kappa=1e-10)
    s = SDynamics(6, 4, lf, kappa=1e-10, metriplectic=False)
    s.G.load_state_dict(sl.G.state_dict())
    z, feat = 0.1 * torch.randn(3, 6, dtype=F64), torch.randn(3, 4, dtype=F64)
    dz_sl, parts = sl(z, feat, raw)
    dz_s, _ = s(z, feat, raw)
    assert torch.allclose(dz_sl - parts["metriplectic"], dz_s, rtol=1e-10, atol=1e-14)


# ---- PREREG F2 (i): conservative minmod reconstruction for arm C ------------------------------
def test_minmod_reconstruction_is_conservative_and_second_order():
    lin = np.linspace(1.0, 2.0, 8)[None, :]
    R = reconstruct_minmod(lin, 4)
    np.testing.assert_allclose(R.reshape(1, 8, 4).mean(-1), lin)                  # mean-preserving
    x = (np.arange(4) + 0.5) / 4 - 0.5
    fine = (lin[0][:, None] + (1 / 7) * x[None, :]).ravel()        # the line through the cell means
    np.testing.assert_allclose(R[0, 4:-4], fine[4:-4], rtol=1e-12)                # linear data exact inside
    peak = np.array([[1.0, 3.0, 1.0, 1.0]])
    R = reconstruct_minmod(peak, 4)
    np.testing.assert_allclose(R[0, 4:8], 3.0)                                    # extremum: flat
    assert R.min() >= peak.min() and R.max() <= peak.max()                        # no new extrema


# ---- Reviewer's disagreement with the Task 3 ruling: the decoder gate on physical fields --------
def test_decoder_gate_uses_physical_fields(tmp_path):
    from p3b.generate import generate_trajectory
    from p3b.latent.autoencoder import decoder_gate
    from p3b.latent.data import load_set
    p = tmp_path / "A.npz"
    assert generate_trajectory("A", POINT_A, "MOF-303", N_z=32, N_c=8, path=p, n_snap=40).ok
    ds = load_set([p])
    std = Standardiser.fit(ds.X, 8, 3, 2)
    phys = ColumnPhysics(setup_a(POINT_A, "MOF-303", 8)[0])

    class LogCOff(torch.nn.Module):          # every log c 0.1 too high: c off by 10.5 %
        def encode(self, x):
            return x

        def decode(self, z):
            X = std.decode(z).clone()
            X[..., 0:24:3] += 0.1
            return std.encode(X)

    g = decoder_gate(LogCOff(), ds, std, phys)
    assert g["field_nrmse_channels"] < 0.02 < g["field_nrmse"] and not g["passed"]


def test_step_jacobian_equals_the_dense_jacobian_inside_no_grad():
    # Newton forms its Jacobian under no_grad; the vectorised reverse-mode one is 4x cheaper than the
    # coloured forward-mode one on CPU (torch.func.vmap over jvp falls back to Python decompositions)
    from p3b.solvers.c_plus_l import step_jacobian
    torch.manual_seed(6)
    col, sched, y0 = _state_a()
    m = Multipliers(n_out=3)
    for net in m.nets:
        net[-1].weight.data.normal_(0, 1.0)
    rhs = TorchColumnRHS(col, m)
    inp = torch.tensor(sched[0][2], dtype=F64)
    dense = torch.autograd.functional.jacobian(lambda x: rhs(x, inp), y0)
    with torch.no_grad():
        J = step_jacobian(lambda x: rhs(x, inp), y0)
    assert not J.requires_grad
    assert torch.allclose(J, dense, rtol=1e-12, atol=1e-12 * float(dense.abs().max()))


def _rollout(rhs, y, inp, h, n, **kw):
    ys = []
    for _ in range(n):
        y = implicit_euler_step(rhs, y, inp, h, **kw)
        ys.append(y)
    return torch.stack(ys)


def test_modified_newton_matches_full_newton_with_fewer_jacobians(monkeypatch):
    # the Jacobian is reused while the residual contracts fast and refreshed when it does not; with no
    # gradient needed the converged point's Jacobian is not formed. The answer is the same to Newton's tol.
    import p3b.solvers.c_plus_l as m
    col, sched, y0 = _state_a()
    rhs = TorchColumnRHS(col, Multipliers(n_out=3, constant=1.0))
    inp, h = torch.tensor(sched[0][2], dtype=F64), sched[0][1] / 100
    calls = {"n": 0}
    real = m.batch_jacobian

    def counted(f, y):
        calls["n"] += 1
        return real(f, y)
    monkeypatch.setattr(m, "batch_jacobian", counted)
    with torch.no_grad():
        full = _rollout(rhs, y0, inp, h, 10, chord=False)
        n_full, calls["n"] = calls["n"], 0
        fast = _rollout(rhs, y0, inp, h, 10)
    sc = full.reshape(10, 3, 32).abs().amax((0, 2), keepdim=True)
    assert float(((fast - full).reshape(10, 3, 32).abs() / sc).max()) < 1e-9
    assert 0 < calls["n"] <= n_full / 2


def test_modified_newton_keeps_the_implicit_function_gradient():
    col, sched, y0 = _state_a()
    m = Multipliers(n_out=3)
    rhs = TorchColumnRHS(col, m)
    inp, h = torch.tensor(sched[0][2], dtype=F64), sched[0][1] / 100
    grads = []
    for chord in (False, True):
        m.zero_grad()
        _rollout(rhs, y0, inp, h, 3, chord=chord)[-1].sum().backward()
        grads.append(torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None]))
    assert torch.allclose(grads[0], grads[1], rtol=1e-7, atol=1e-12 * float(grads[0].abs().max()))
