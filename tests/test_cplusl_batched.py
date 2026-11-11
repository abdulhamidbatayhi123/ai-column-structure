"""C+L batched over trajectories (one Newton solve for a batch of windows or trajectories): every
batched result equals the per-trajectory one, so batching changes cost, never results."""
import numpy as np
import pytest
import torch

from p3b.simulate_a import prewet_state, setup_a
from p3b.solvers.c_plus_l import Multipliers, TorchColumnRHS, implicit_euler_step

F64 = torch.float64
POINTS = [dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=v, alpha=0.8, beta=0.6, index=i, n_cycles=1)
          for i, v in enumerate((0.05, 0.10, 0.20))]


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
    return float(((a - b).reshape(B, -1, N).abs() / sc).max())


def test_stacked_rhs_equals_each_trajectory_rhs():
    m = _mult()
    rhs, Y, I, _ = _batch(m)
    st = TorchColumnRHS.stack(rhs)
    out = st(Y, I)
    ref = torch.stack([r(Y[j], I[j]) for j, r in enumerate(rhs)])
    assert out.shape == ref.shape
    assert _rel(out, ref) < 1e-13


def test_stack_refuses_different_multipliers_or_materials():
    rhs, _, _, _ = _batch(_mult())
    other = TorchColumnRHS(setup_a(POINTS[0], "MOF-303", 32)[0], _mult(7))
    with pytest.raises(ValueError):
        TorchColumnRHS.stack(rhs + [other])
    with pytest.raises(ValueError):
        TorchColumnRHS.stack(rhs + [TorchColumnRHS(setup_a(POINTS[0], "Al-fumarate", 32)[0], rhs[0].mult)])


def test_batched_implicit_euler_equals_each_trajectory():
    m = _mult()
    rhs, Y, I, H = _batch(m)
    st = TorchColumnRHS.stack(rhs)
    with torch.no_grad():
        yb, ys = Y, [Y[j] for j in range(3)]
        for _ in range(5):
            yb = implicit_euler_step(st, yb, I, H, substeps=2)
            ys = [implicit_euler_step(r, ys[j], I[j], float(H[j]), substeps=2) for j, r in enumerate(rhs)]
    assert _rel(yb, torch.stack(ys)) < 1e-9


def test_batched_gradient_equals_the_sum_of_each_trajectory_gradient():
    m = _mult()
    rhs, Y, I, H = _batch(m)
    st = TorchColumnRHS.stack(rhs)
    m.zero_grad()
    implicit_euler_step(st, implicit_euler_step(st, Y, I, H), I, H).pow(2).sum().backward()
    gb = torch.cat([p.grad.flatten() for p in m.parameters()])
    m.zero_grad()
    for j, r in enumerate(rhs):
        implicit_euler_step(r, implicit_euler_step(r, Y[j], I[j], float(H[j])), I[j], float(H[j])).pow(2).sum().backward()
    gs = torch.cat([p.grad.flatten() for p in m.parameters()])
    assert torch.allclose(gb, gs, rtol=1e-6, atol=1e-9 * float(gs.abs().max()))


def test_a_failed_newton_solve_in_a_batch_raises():
    m = Multipliers(n_out=3, constant=1.0)
    rhs, Y, I, H = _batch(m)
    st = TorchColumnRHS.stack(rhs)
    Y = Y.clone()
    Y[1, :32] = float("nan")
    with torch.no_grad(), pytest.raises(RuntimeError):
        implicit_euler_step(st, Y, I, H)


# ---- training loss, validation loss and evaluation rollouts on a batch ---------------------------
@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    from p3b.generate import generate_trajectory
    d = tmp_path_factory.mktemp("clb")
    out = []
    for pt in POINTS:
        p = d / f"A{pt['index']}.npz"
        assert generate_trajectory("A", pt, "MOF-303", N_z=32, N_c=8, path=p, n_snap=20).ok
        out.append(p)
    return out


def _serial_loss(data, n_windows, n_steps, K, rng):
    """The pre-batching loss: windows one after another (the reference)."""
    total = 0.0
    for _ in range(n_windows):
        j = int(rng.integers(len(data.Y)))
        Y = data.Y[j].reshape(-1, data.Y[j].shape[-1])
        s = int(rng.integers(Y.shape[0] - n_steps))
        y = Y[s]
        for k in range(n_steps):
            i = s + k
            y = implicit_euler_step(data.rhs[j], y, data.raw[j][i, :3], float(data.t[j][i + 1] - data.t[j][i]),
                                    substeps=K)
            total = total + (((y - Y[i + 1]) / data.sd) ** 2).mean()
    return total / (n_windows * n_steps)


def test_batched_training_loss_and_gradient_equal_the_serial_ones(paths):
    from p3b.latent.data import load_set
    from p3b.train import cplusl
    m = _mult(sd=0.05)
    data = cplusl._Data(load_set(paths), "MOF-303", 8, m)
    m.zero_grad()
    ref = _serial_loss(data, 4, 3, 2, np.random.default_rng(5))
    ref.backward()
    g_ref = torch.cat([p.grad.flatten() for p in m.parameters()])
    m.zero_grad()
    got = cplusl._loss(data, 4, 3, 2, np.random.default_rng(5))
    got.backward()
    g = torch.cat([p.grad.flatten() for p in m.parameters()])
    assert float(got) == pytest.approx(float(ref), rel=1e-8)
    assert torch.allclose(g, g_ref, rtol=1e-6, atol=1e-9 * float(g_ref.abs().max()))


def test_batched_rollouts_equal_single_rollouts(paths, tmp_path):
    from p3b.train.cplusl import rollout_cplusl_to_file, rollout_cplusl_to_files
    m = Multipliers(n_out=3, constant=1.0)
    single = [tmp_path / f"s{j}.npz" for j in range(3)]
    for pt, p in zip(POINTS, single):
        rollout_cplusl_to_file(m, pt, "MOF-303", n_coarse=4, n_target=8, n_snap=20, K=8, path=p)
    batch = [tmp_path / f"b{j}.npz" for j in range(3)]
    rollout_cplusl_to_files(m, POINTS, "MOF-303", n_coarse=4, n_target=8, n_snap=20, K=8, paths=batch)
    for a, b in zip(single, batch):
        A, B = np.load(a), np.load(b)
        assert set(A.files) == set(B.files)
        for k in A.files:
            if A[k].dtype.kind == "f":
                sc = max(float(np.abs(A[k]).max()), 1e-300)
                assert float(np.abs(A[k] - B[k]).max()) <= 1e-8 * sc, k
            else:
                assert (A[k] == B[k]).all(), k


def test_batched_validation_loss_equals_the_serial_one(paths):
    from p3b.latent.data import load_set
    from p3b.train import cplusl
    m = Multipliers(n_out=3, constant=1.0)
    val = load_set(paths[1:])
    data = cplusl._Data(val, "MOF-303", 8, m)
    ns = val.X.shape[2]
    ref = []
    with torch.no_grad():
        for j in range(len(data.Y)):                 # the pre-batching loop (the reference)
            Y = data.Y[j].reshape(-1, data.Y[j].shape[-1])
            y, tot = Y[0], 0.0
            for i in range(ns - 1):
                y = implicit_euler_step(data.rhs[j], y, data.raw[j][i, :3],
                                        float(data.t[j][i + 1] - data.t[j][i]), substeps=2)
                tot += float((((y - Y[i + 1]) / data.sd) ** 2).mean())
            ref.append(tot / (ns - 1))
    got = cplusl.cplusl_val_loss(m, val, "MOF-303", n_coarse=8, K=2)
    assert np.isfinite(got)
    assert got == pytest.approx(float(np.mean(ref)), rel=1e-8)


def test_a_failing_trajectory_fails_alone_and_the_others_are_stored(paths, tmp_path, monkeypatch):
    # evaluation scores a Newton failure as that trajectory's failure: a batch that fails falls back
    # to one rollout per trajectory, so the outcome is the per-trajectory one
    from p3b.train import cplusl
    m = Multipliers(n_out=3, constant=1.0)
    real = cplusl.rollout_cplusl_to_files

    def flaky(mult, points, *a, paths, **k):
        if any(pt["index"] == 1 for pt in points):
            raise RuntimeError("Newton did not converge")
        return real(mult, points, *a, paths=paths, **k)
    monkeypatch.setattr(cplusl, "rollout_cplusl_to_files", flaky)
    out = cplusl.rollout_cplusl_each(m, POINTS, "MOF-303", n_coarse=4, n_target=8, n_snap=20, K=8,
                                     paths=[tmp_path / f"e{j}.npz" for j in range(3)])
    assert out[1] is None and out[0] is not None and out[2] is not None
    single = tmp_path / "s2.npz"
    real(m, [POINTS[2]], "MOF-303", 4, 8, 20, 8, paths=[single])
    assert np.allclose(np.load(out[2])["channels"], np.load(single)["channels"], rtol=1e-8, atol=0)
