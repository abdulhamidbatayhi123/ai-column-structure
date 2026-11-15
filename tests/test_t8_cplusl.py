"""T8 review, the C+L solver and the solver arms' scoring (findings CL-1 to CL-7).

CL-1: a batched Newton solve latches each sample when it is accepted and restarts each sample that
goes astray, so a batch succeeds exactly when every sample alone does. CL-2: a solver run that fails
in cycle k keeps its cycles before k and is capped from cycle k on (PREREG 5). CL-3: a chord solve
that fails is retried with full Newton before the failure counts. CL-4: Newton failures are their
own exception class, their reason is kept, and every other error propagates. CL-5: a stall-accepted
solve without a gradient takes the final Newton correction. CL-6 and CL-7: documentation."""
import json
import os

import numpy as np
import pytest
import torch

import p3b.solvers.c_plus_l as cl
from p3b.simulate_a import prewet_state, setup_a
from p3b.solvers.c_plus_l import Multipliers, TorchColumnRHS, implicit_euler_step

F64 = torch.float64
NewtonFailure = getattr(cl, "NewtonFailure", RuntimeError)     # before the fix: the bare RuntimeError
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POINTS = [dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=v, alpha=0.8, beta=0.6, index=i, n_cycles=1)
          for i, v in enumerate((0.05, 0.10, 0.20))]


# ---- a scripted right-hand side: the test sets every Newton residual ----------------------------
class _Scripted:
    """y' = -y + p with one cell and one variable per sample. p is scripted per sample and per
    residual evaluation (call 0 is the scale, call k + 1 Newton's iteration k); Jacobian passes see
    p = 0, so the Jacobian is exactly -1. With y0 = 1 and h = 1, Newton's residual after a step taken
    at iteration k is p_k - p_{k+1}: the script sets each sample's residual sequence exactly.
    inp[..., 0] is the sample's index into the script."""
    N = 1

    def __init__(self, P):
        self.P, self.calls = P, 0

    @classmethod
    def from_residuals(cls, rels):
        """rels[s]: sample s's residuals at iterations 1, 2, ... (iteration 0's is 1)."""
        P = []
        for r in rels:
            p, sign = [0.0, 0.0], 1.0
            for x in r:
                p.append(p[-1] + sign * x)
                sign = -sign
            P.append(p)
        return cls(P)

    def __call__(self, x, inp):
        if x.requires_grad:                          # a Jacobian pass
            return -x
        c = self.calls
        self.calls += 1
        ids = inp[..., 0].reshape(-1).long()
        p = torch.tensor([self.P[int(i)][min(c, len(self.P[int(i)]) - 1)] for i in ids], dtype=F64)
        return -x + p.reshape(x.shape)


def _solve(P, ids, **kw):
    """The batch `ids` in one solve (a 1-D state for one sample), y0 = 1, h = 1."""
    rhs = _Scripted(P)
    if len(ids) == 1:
        y0, inp = torch.ones(1, dtype=F64), torch.tensor([float(ids[0])], dtype=F64)
    else:
        y0, inp = torch.ones(len(ids), 1, dtype=F64), torch.tensor([[float(i)] for i in ids], dtype=F64)
    with torch.no_grad():
        return implicit_euler_step(rhs, y0, inp, 1.0, **kw)


# Sample 0 alone is stall-accepted at iteration 3 (30e-10 twice after a refresh) and then halves above
# tol; sample 1 alone is stall-accepted at iteration 5. Together, without a per-sample latch, one of
# them is always halving above tol when the other stalls, so the batch never finishes.
STALL_SCRIPT = [[30e-10, 30e-10, 30e-10, 14e-10, 6.5e-10, 3e-10, 1.4e-10, 1.2e-10],
                [30e-10, 30e-10, 14e-10, 14e-10, 14e-10, 14e-10, 14e-10, 14e-10]]


def test_cl1_a_batch_succeeds_when_each_sample_is_stall_accepted_at_a_different_iteration():
    P = _Scripted.from_residuals(STALL_SCRIPT).P
    alone = [_solve(P, [j], max_iter=8) for j in (0, 1)]          # each converges by the stall rule
    together = _solve(P, [0, 1], max_iter=8)                       # raised before the per-sample latch
    assert together.shape == (2, 1)
    for j in (0, 1):
        assert abs(float(together[j, 0]) - float(alone[j][0])) < 1e-15


def test_cl1_each_astray_sample_restarts_once_on_its_own():
    # a carried Jacobian (the right one, 2 = 1 - h J) and a non-finite residual: sample 0 at
    # iteration 0, sample 1 later after a chord step. Alone each restarts and converges; the batch
    # restarted once for everyone, so sample 1's later NaN failed it.
    nan = float("nan")
    P = [[0.0, nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.3, nan, 0.0, 0.0, 0.0]]

    def cache(B):
        return {"B": B, "hs": torch.ones(B, 1, dtype=F64),
                "lu": torch.linalg.lu_factor(2.0 * torch.ones(B, 1, 1, dtype=F64))}
    alone = [_solve(P, [j], jac_cache=cache(1)) for j in (0, 1)]
    together = _solve(P, [0, 1], jac_cache=cache(2))
    for j in (0, 1):
        assert float(alone[j][0]) == pytest.approx(0.5, abs=1e-15)
        assert float(together[j, 0]) == pytest.approx(0.5, abs=1e-15)


def _mult(seed=6, sd=0.3):
    torch.manual_seed(seed)
    m = Multipliers(n_out=3)
    for net in m.nets:
        net[-1].weight.data.normal_(0, sd)
    return m


def _batch(m, N=32):
    rhs, ys, inps, hs = [], [], [], []
    for pt in POINTS:
        col, sched = setup_a(pt, "MOF-303", N)
        rhs.append(TorchColumnRHS(col, m))
        ys.append(torch.tensor(prewet_state(col, sched[0][2][0], pt["T_ads"]), dtype=F64))
        inps.append(torch.tensor(sched[0][2], dtype=F64))
        hs.append(sched[0][1] / 100)
    return rhs, torch.stack(ys), torch.stack(inps), torch.tensor(hs, dtype=F64)


def _rel(a, b, N=32):
    B = a.shape[0]
    sc = b.reshape(B, -1, N).abs().amax(-1, keepdim=True).clamp(min=1e-300)
    return float(((a - b).detach().reshape(B, -1, N).abs() / sc.detach()).max())


def _try(fn):
    try:
        return fn()
    except NewtonFailure:
        return None


@pytest.mark.parametrize("unit", [True, False])
def test_cl1_batched_and_single_solves_agree_when_every_exit_is_by_the_stall_rule(unit):
    # tol below round-off: every sample must leave by the stall rule; the batch must succeed exactly
    # when every single solve does, and agree with them to the stall tolerance's level
    m = Multipliers(n_out=3, constant=1.0) if unit else _mult()
    rhs, Y, I, H = _batch(m)
    st = TorchColumnRHS.stack(rhs)
    with torch.no_grad():
        yb, ys = Y, [Y[j] for j in range(3)]
        for _ in range(3):
            yb = _try(lambda: implicit_euler_step(st, yb, I, H, tol=1e-15))
            ys = [_try(lambda j=j, r=r: implicit_euler_step(r, ys[j], I[j], float(H[j]), tol=1e-15))
                  for j, r in enumerate(rhs)]
            assert (yb is None) == any(y is None for y in ys)
            if yb is None:
                return
            assert _rel(yb, torch.stack(ys)) < 1e-7


# ---- CL-3: full Newton before a failure counts -------------------------------------------------
def _state_a(N=32):
    pt = dict(POINTS[1])
    col, sched = setup_a(pt, "MOF-303", N)
    z = (np.arange(N) + 0.5) / N
    c_in = sched[0][2][0]
    c = c_in * (0.05 + 0.9 / (1 + np.exp((z - 0.4) / 0.06)))
    T = 303.15 + 12 * np.exp(-((z - 0.5) / 0.2) ** 2)
    n, a = col.eps * c, col.A * 0.9 * col.q_star(c, T)
    return col, sched, torch.tensor(np.concatenate([n, a, col.energy(n, a, T)]), dtype=F64)


def test_cl3_a_stale_carried_jacobian_does_not_fail_a_solve_that_full_newton_converges():
    col, sched, y0 = _state_a()
    rhs = TorchColumnRHS(col, Multipliers(n_out=3, constant=1.0))
    inp, h = torch.tensor(sched[0][2], dtype=F64), sched[0][1] / 100
    with torch.no_grad():
        m = next(k for k in range(1, 40) if _try(lambda: implicit_euler_step(rhs, y0, inp, h, chord=False,
                                                                              max_iter=k)) is not None)
        assert m >= 2
        full = implicit_euler_step(rhs, y0, inp, h, chord=False, max_iter=m)
        n = y0.numel()
        stale = {"B": 1, "hs": torch.full((1, 1), h, dtype=F64),       # a factorisation far from this state's
                 "lu": torch.linalg.lu_factor(1e12 * torch.eye(n, dtype=F64).unsqueeze(0))}
        got = implicit_euler_step(rhs, y0, inp, h, max_iter=m, jac_cache=stale)
    assert _rel(got.unsqueeze(0), full.unsqueeze(0)) < 1e-9
    assert stale.get("full_newton_retries") == 1


def test_cl3_a_solve_that_full_newton_cannot_converge_still_fails():
    col, sched, y0 = _state_a()
    rhs = TorchColumnRHS(col, Multipliers(n_out=3, constant=1.0))
    with torch.no_grad(), pytest.raises(NewtonFailure, match="Newton"):
        implicit_euler_step(rhs, y0, torch.tensor(sched[0][2], dtype=F64), 50.0, max_iter=1)


# ---- CL-4: a dedicated exception, its reason kept, other errors propagate -----------------------
def test_cl4_newton_failure_is_its_own_runtime_error_subclass():
    assert hasattr(cl, "NewtonFailure") and issubclass(cl.NewtonFailure, RuntimeError)
    assert cl.NewtonFailure is not RuntimeError
    col, sched, y0 = _state_a()
    rhs = TorchColumnRHS(col)
    with torch.no_grad(), pytest.raises(cl.NewtonFailure):
        implicit_euler_step(rhs, y0, torch.tensor(sched[0][2], dtype=F64), 50.0, max_iter=1)


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    from p3b.generate import generate_trajectory
    d = tmp_path_factory.mktemp("t8cl")
    out = []
    for pt in POINTS:
        p = d / f"A{pt['index']}.npz"
        assert generate_trajectory("A", pt, "MOF-303", N_z=32, N_c=8, path=p, n_snap=20).ok
        out.append(p)
    return out


def test_cl4_an_error_that_is_not_a_newton_failure_propagates(paths, tmp_path, monkeypatch):
    from p3b.latent.data import load_set
    from p3b.train import cplusl

    def broken(*a, **k):
        raise RuntimeError("shape mismatch")       # a bug, not a solver failure
    monkeypatch.setattr(cplusl, "implicit_euler_step", broken)
    ds = load_set(paths)
    m = Multipliers(n_out=3, constant=1.0)
    with pytest.raises(RuntimeError, match="shape"):
        cplusl.train_cplusl(Multipliers(n_out=3), ds, "MOF-303", n_coarse=4, lr=1e-3, iterations=1, seed=0,
                            n_windows=1, n_steps=2)
    with pytest.raises(RuntimeError, match="shape"):
        cplusl.cplusl_val_loss(m, ds, "MOF-303", n_coarse=4, K=1)
    with pytest.raises(RuntimeError, match="shape"):
        cplusl.rollout_cplusl_each(m, POINTS, "MOF-303", n_coarse=4, n_target=8, n_snap=20, K=8,
                                   paths=[tmp_path / f"e{j}.npz" for j in range(3)])


def test_cl4_rollouts_record_the_failure_reason_and_the_fallback(tmp_path, monkeypatch):
    from p3b.train import cplusl
    real = cplusl.implicit_euler_step

    def planted(rhs, y0, *a, **k):
        if y0.shape[0] > 1 or float(rhs.v.reshape(-1)[0]) == POINTS[1]["v"]:
            raise NewtonFailure("planted non-convergence")
        return real(rhs, y0, *a, **k)
    monkeypatch.setattr(cplusl, "implicit_euler_step", planted)
    report = {}
    out = cplusl.rollout_cplusl_each(Multipliers(n_out=3, constant=1.0), POINTS, "MOF-303", n_coarse=4,
                                     n_target=8, n_snap=20, K=8,
                                     paths=[tmp_path / f"e{j}.npz" for j in range(3)], report=report)
    assert out[1] is None and out[0] is not None and out[2] is not None
    assert len(report["batch_fallback"]) == 1 and "planted" in report["batch_fallback"][0]["reason"]
    assert report["batch_fallback"][0]["trajectories"] == [0, 1, 2]
    assert set(report["failures"]) == {1}
    assert report["failures"][1]["failed_cycle"] == 1 and "planted" in report["failures"][1]["reason"]


def test_cl4_training_keeps_the_reason_of_each_skipped_iteration(paths, monkeypatch):
    from p3b.latent.data import load_set
    from p3b.train import cplusl

    def planted(*a, **k):
        raise NewtonFailure("planted non-convergence")
    monkeypatch.setattr(cplusl, "implicit_euler_step", planted)
    h = cplusl.train_cplusl(Multipliers(n_out=3), load_set(paths), "MOF-303", n_coarse=4, lr=1e-3, iterations=2,
                            seed=0, n_windows=1, n_steps=2)
    assert h["skipped"] == 2
    assert [it for it, _ in h["newton_failures"]] == [0, 1]
    assert all("planted" in msg for _, msg in h["newton_failures"])


# ---- CL-2: a failure in cycle k is scored from cycle k on ---------------------------------------
@pytest.fixture(scope="module")
def four_cycles(tmp_path_factory):
    from p3b.generate import generate_trajectory
    d = tmp_path_factory.mktemp("t8four")
    (d / "validation").mkdir()
    pt = dict(POINTS[1], n_cycles=4, index=0)
    assert generate_trajectory("A", pt, "MOF-303", N_z=32, N_c=8, path=d / "validation" / "t0.npz", n_snap=20).ok
    return d, pt


def _check_failed_from(row, k):
    """Cycles before k are scored from the arm's own prediction (capped only where it is itself
    beyond the cap); cycles from k on are capped."""
    from p3b.eval.metrics import CAP
    first = next((i + 1 for i in range(k - 1)
                  if row["e_uncapped"][i] > CAP or row["eps_uncapped"][i] > CAP), k)
    assert row["failed_cycle"] == first
    for i in range(first - 1):
        assert row["e"][i] == row["e_uncapped"][i] > 0      # a real prediction, not the reference
        assert row["eps"][i] == row["eps_uncapped"][i]
    assert all(x == CAP for x in row["e"][first - 1:]) and all(x == CAP for x in row["eps"][first - 1:])


def _fail_in_cycle(monkeypatch, k, n_snap=20):
    """implicit_euler_step raises a Newton failure from the first interval of cycle k (n_snap
    intervals per cycle)."""
    from p3b.train import cplusl
    real = cplusl.implicit_euler_step
    count = {"n": 0}

    def failing(*a, **kw):
        count["n"] += 1
        if count["n"] > (k - 1) * n_snap:
            raise NewtonFailure(f"planted failure in cycle {k}")
        return real(*a, **kw)
    monkeypatch.setattr(cplusl, "implicit_euler_step", failing)


def test_cl2_a_cplusl_failure_in_cycle_3_keeps_cycles_1_and_2(four_cycles, tmp_path, monkeypatch):
    from p3b.eval.evaluate import evaluate_files
    from p3b.latent.data import load_set
    from p3b.latent.physics import ColumnPhysics
    from p3b.train import cplusl
    d, pt = four_cycles
    _fail_in_cycle(monkeypatch, 3)
    report = {}
    out = cplusl.rollout_cplusl_each(Multipliers(n_out=3, constant=1.0), [pt], "MOF-303", n_coarse=4, n_target=8,
                                     n_snap=20, K=8, paths=[tmp_path / "cl.npz"], report=report)
    assert out[0] is not None
    assert np.load(out[0])["channels"].shape[0] == 2                   # cycles 1 and 2 stored
    assert report["failures"][0]["failed_cycle"] == 3
    ds = load_set([d / "validation" / "t0.npz"])
    phys = ColumnPhysics(setup_a(pt, "MOF-303", 8)[0])
    row, = evaluate_files({"C+L": out}, ds, phys, n_cycles=4, seed=0)
    _check_failed_from(row, 3)


def test_cl2_a_c_failure_in_cycle_3_keeps_cycles_1_and_2(four_cycles, tmp_path, monkeypatch):
    from p3b.solvers import coarse
    from p3b.train.cells import Cell, train_cell
    d, _ = four_cycles
    real = coarse.simulate_trajectory_a

    def failing(point, material, N_z, n_snap=200, on_cycle=None, **kw):
        def on(k, t, y, p):
            if k == 2:
                raise RuntimeError("planted failure in cycle 3")
            on_cycle(k, t, y, p)
        return real(point, material, N_z, n_snap=n_snap, on_cycle=on, **kw)
    monkeypatch.setattr(coarse, "simulate_trajectory_a", failing)
    pilot = {"floors": {"N": 1e-6, "E": 1e-6}, "tau_S": 1e-6}
    out = train_cell(Cell("C", "A", "MOF-303", "full", 0), str(d), str(tmp_path), pilot, quick=dict(grids=[8]))
    row, = json.loads(out.read_text())["rows"]
    _check_failed_from(row, 3)
    assert "planted failure in cycle 3" in row["failure_reason"]


def test_cl2_a_cplusl_cell_scores_a_cycle_3_failure_from_cycle_3_and_records_why(four_cycles, tmp_path,
                                                                                  monkeypatch):
    from p3b.train.cells import Cell, _solver_cell, _weights_file
    d, _ = four_cycles
    cell = Cell("C+L", "A", "MOF-303", "full", 0)
    torch.save(Multipliers(n_out=3).state_dict(), tmp_path / _weights_file(cell))   # every multiplier 1
    _fail_in_cycle(monkeypatch, 3)
    pilot = {"K": 8, "floors": {"N": 1e-6, "E": 1e-6}, "tau_S": 1e-6}
    out = _solver_cell(cell, str(d), str(tmp_path), pilot, {"n_coarse": 4}, ("validation",), train=False)
    doc = json.loads(out.read_text())
    row, = doc["rows"]
    _check_failed_from(row, 3)
    assert "planted failure in cycle 3" in row["failure_reason"]
    rep = doc["meta"]["rollout_report"]["validation"]
    assert rep["failures"]["0"]["failed_cycle"] == 3 and rep["batch_fallback"] == []


# ---- CL-5: the stall-accepted value takes the final Newton correction ---------------------------
def test_cl5_a_stall_accepted_solve_returns_the_corrected_value():
    # sample 0 of STALL_SCRIPT is accepted by the stall rule at iteration 3; full Newton's final
    # correction there (the Jacobian is exact and constant) moves it to (1 + p_3) / 2
    P = _Scripted.from_residuals(STALL_SCRIPT).P
    got = float(_solve(P, [0], max_iter=8)[0])
    assert got == pytest.approx((1 + P[0][4]) / 2, abs=1e-13)
    assert abs(got - (1 + P[0][3]) / 2) > 1e-9                     # not the uncorrected iterate


def test_cl5_stalled_solves_without_a_gradient_agree_with_full_newton_to_tol():
    col, sched, y0 = _state_a()
    rhs = TorchColumnRHS(col, _mult())
    inp, h = torch.tensor(sched[0][2], dtype=F64), sched[0][1] / 100
    with torch.no_grad():
        fast = _try(lambda: implicit_euler_step(rhs, y0, inp, h, tol=1e-15))
        full = _try(lambda: implicit_euler_step(rhs, y0, inp, h, tol=1e-15, chord=False))
    assert (fast is None) == (full is None)
    if fast is not None:
        assert _rel(fast.unsqueeze(0), full.unsqueeze(0)) < 1e-9


# ---- CL-6 and CL-7: documentation -------------------------------------------------------------
def test_cl6_the_solver_docstrings_name_the_jacobian_newton_uses():
    doc = implicit_euler_step.__doc__
    assert "batch_jacobian" in doc and "step_jacobian" not in doc
    assert "Newton's Jacobian" not in cl.step_jacobian.__doc__


def test_cl6_coloured_jacobian_is_exact_for_system_b_with_learned_multipliers():
    from p3b.simulate_b import T_AMB, make_store
    from p3b.solvers.c_plus_l import TorchStoreRHS, batch_jacobian
    N = 16
    st = make_store(N)
    torch.manual_seed(4)
    m = Multipliers(n_out=2)
    for net in m.nets:
        net[-1].weight.data.normal_(0, 1.0)
    rhs = TorchStoreRHS(st, m)
    z = (np.arange(N) + 0.5) / N
    Tf = 300 + 50 / (1 + np.exp((z - 0.5) / 0.1))
    y = torch.tensor(np.concatenate([st.Cf * (Tf - 293.15), st.Cs * (Tf - 1 - 293.15)]), dtype=F64)
    for v in (2.0e-3, -2.0e-3):
        inp = torch.tensor([v, 353.15 if v > 0 else 298.15, T_AMB], dtype=F64)
        with torch.no_grad():
            J = batch_jacobian(lambda x: rhs(x[0], inp).unsqueeze(0), y.unsqueeze(0), N)[0]
        dense = torch.autograd.functional.jacobian(lambda x: rhs(x, inp), y)
        assert torch.allclose(J, dense, rtol=1e-12, atol=1e-12 * float(dense.abs().max()))


def test_cl7_each_training_timing_kernel_names_the_kit_it_ran_on():
    def read(name):
        with open(os.path.join(ROOT, "results", name)) as f:
            return f.read()
    cur = read("timing_cplusl_train_kernel.py")
    assert "9dc6103" in cur.splitlines()[0] and "212ba17" not in cur and "for K in (1, 4)" in cur
    assert json.loads(read("timing_cplusl_train.json"))["kit"].startswith("9dc6103")
    old = read("timing_cplusl_train_212ba17_kernel.py")
    assert "212ba17" in old.splitlines()[0] and "for K in (1, 2)" in old
    assert json.loads(read("timing_cplusl_train_212ba17.json"))["kit"].startswith("212ba17")
