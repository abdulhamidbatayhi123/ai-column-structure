"""Batched rollouts (2026-11-10): a test set advanced in one tensor, each trajectory's result as if alone.

Kaggle: S's step for 48 trajectories costs 2.5 x one (results/timing_concurrency_batch.json), so the
evaluation and the pilot's validation rollouts batch. `rollout_batch(arm, ds, js, ...)` must equal
`[rollout(arm, ds, j, ...) for j in js]` up to rounding, failures included: a trajectory that fails
leaves the batch, the others go on.
"""
import pytest
import torch
from torch import nn

from p3b.eval.rollout import rollout, rollout_batch
from p3b.generate import generate_trajectory
from p3b.latent.data import Standardiser, load_set
from p3b.latent.integrate import balance_project
from p3b.train.trainer import Arm

F64 = torch.float64
NC, NS = 8, 40
POINTS = [dict(RH_ads=0.30 + 0.08 * i, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.7 + 0.1 * i, beta=0.6,
               index=i, n_cycles=2) for i in range(3)]


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    d = tmp_path_factory.mktemp("rob")
    paths = []
    for i, pt in enumerate(POINTS):
        p = d / f"t{i}.npz"
        assert generate_trajectory("A", pt, "MOF-303", N_z=32, N_c=NC, path=p, n_snap=NS).ok
        paths.append(p)
    return load_set(paths)


class Identity(nn.Module):
    def encode(self, x):
        return x

    def decode(self, z):
        return z


class Smooth(nn.Module):
    """A fixed, non-trivial, input-dependent vector field."""

    def __init__(self, n, k):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.W = nn.Parameter(1e-3 * torch.randn(n, n, generator=g, dtype=F64))
        self.V = nn.Parameter(1e-3 * torch.randn(k, n, generator=g, dtype=F64))

    def forward(self, z, feat):
        return torch.tanh(z @ self.W + feat @ self.V)


def _same(a, b):
    assert a.failed_cycle == b.failed_cycle and a.error == b.error
    assert torch.equal(a.finite, b.finite)
    assert torch.allclose(a.X, b.X, rtol=1e-11, atol=0, equal_nan=True)
    assert torch.allclose(a.X_end, b.X_end, rtol=1e-11, atol=0, equal_nan=True)
    assert a.flags == b.flags
    assert set(a.residuals) == set(b.residuals)
    for k in a.residuals:
        assert torch.allclose(a.residuals[k], b.residuals[k], rtol=1e-8, atol=1e-12 * a.residuals[k].abs().max())


def test_batched_rollout_equals_single_rollouts(ds):
    std = Standardiser.fit(ds.X, NC, 3, 2, features=ds.features)
    arm = Arm("U", Smooth(ds.X.shape[-1], ds.features.shape[-1]), Identity(), std, None)
    batch = rollout_batch(arm, ds, [0, 1, 2], n_cycles=2, h_steps=2)
    for j, r in zip([0, 1, 2], batch):
        _same(r, rollout(arm, ds, j, n_cycles=2, h_steps=2))
    assert all(r.failed_cycle is None for r in batch)


class MarkedArm(Arm):
    """Arm whose step breaks the trajectory with the largest c_in (index 2) during adsorption:
    mode "nan" returns a non-finite state, mode "linalg" raises as a diverged projection would."""

    def __init__(self, *a, mode="nan", marker=None, **k):
        super().__init__(*a, **k)
        self.mode, self.marker = mode, marker

    def step(self, z, feat, raw, dt, h_steps, track=False):
        bad = raw[:, 0] > self.marker
        if bool(bad.any()) and self.mode == "linalg":
            raise torch.linalg.LinAlgError("diverged (test)")
        z, info = super().step(z, feat, raw, dt, h_steps, track)
        if bool(bad.any()):
            z = z.clone()
            z[bad] = float("nan")                      # only the marked trajectory; the others advance
        return z, info


@pytest.mark.parametrize("mode", ["nan", "linalg"])
def test_a_failing_trajectory_leaves_the_batch_and_the_others_go_on(ds, mode):
    std = Standardiser.fit(ds.X, NC, 3, 2, features=ds.features)
    c_ads = ds.raw_inputs[:, 0, 0, 0]                        # feed c_in of the first interval, per trajectory
    marker = float((c_ads[1] + c_ads[2]) / 2)
    assert float(c_ads[2]) > marker > float(c_ads[1])
    arm = MarkedArm("U", Smooth(ds.X.shape[-1], ds.features.shape[-1]), Identity(), std, None,
                    mode=mode, marker=marker)
    batch = rollout_batch(arm, ds, [0, 1, 2], n_cycles=2, h_steps=1)
    singles = [rollout(arm, ds, j, n_cycles=2, h_steps=1) for j in (0, 1, 2)]
    for a, b in zip(batch, singles):
        _same(a, b)
    assert batch[2].failed_cycle == 1 and batch[0].failed_cycle is None and batch[1].failed_cycle is None
    if mode == "linalg":
        assert "diverged (test)" in batch[2].error


def test_projection_leaves_a_converged_sample_alone_while_others_iterate():
    """Each sample's projection is its own: a sample within tolerance gets no further update while its
    batch mates iterate, so its result does not depend on the batch it is in."""
    def cons(x):
        return torch.stack([x.square().sum(-1), x[..., 0] * x[..., 1]], -1)
    z = torch.tensor([[0.5, 0.4, 0.3], [0.9, -0.2, 0.7]], dtype=F64)
    target = cons(z)
    target[0] = target[0] * (1 + 1e-13)                     # sample 0: inside the 1e-12 tolerance already
    target[1] = target[1] * 1.3                             # sample 1: needs several updates
    zs, info = balance_project(z, cons, target)
    assert bool(info["converged"].all())
    assert torch.equal(zs[0], z[0])
    alone, _ = balance_project(z[1:], cons, target[1:])
    assert torch.allclose(zs[1], alone[0], rtol=1e-14, atol=0)


def test_batched_rollout_of_a_projected_arm_equals_single_rollouts(tmp_path):
    """U+proj through the real pilot builders: projection, its flags and the tracked balance residuals.
    (The quick S model diverges in cycle 1 at 8 cells, where a 1e-16 change of the initial state decides
    the failing interval, so S's per-sample flags are checked one step at a time below.)"""
    import importlib.util
    import pathlib
    from p3b.pilot import pieces
    root = pathlib.Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("time_arms", root / "scripts" / "time_arms.py")
    ta = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ta)
    ta.make_data(str(tmp_path), 32, 8, 40)
    q = {"d": 4, "ae_epochs": 30, "ae_width": 16, "width": 8, "n_windows": 1, "n_steps": 2}
    key = ta.KEY
    train = load_set(pieces.files(str(tmp_path), key, "train"))
    out = tmp_path / "ae"
    out.mkdir()
    st = {"d": {key: 4}}
    pieces.piece_ae(str(tmp_path), key, 0, st, str(out), q)
    s_arm = pieces.make_arm("S", key, 0, st, train, str(out), q)
    u = pieces.make_arm("U", key, 0, st, train, str(out), q)
    u_proj = Arm("U+proj", u.dyn, u.ae, u.std, s_arm.lf)       # U's weights, S's projection at rollout
    batch = rollout_batch(u_proj, train, [0, 1], n_cycles=2, h_steps=2, track=True)
    # This untrained model's projection does not converge in its first interval (balance residual ~1e4
    # after 20 Gauss-Newton updates), so batched and single linear algebra differ there at ~1e-10 and the
    # toy dynamics amplify it later (measured: 5e-2 by snapshot 8; U alone stays at 8e-17 throughout).
    # Checked here: the first interval, and that the same flags and residuals are kept.
    for j, r in zip([0, 1], batch):
        a = rollout(u_proj, train, j, n_cycles=2, h_steps=2, track=True)
        x_b, x_a = r.X.reshape(-1, r.X.shape[-1])[:2], a.X.reshape(-1, a.X.shape[-1])[:2]
        assert torch.allclose(x_b, x_a, rtol=1e-8, atol=0)
        assert r.failed_cycle == a.failed_cycle is None
        assert set(r.flags) == set(a.flags) and "projection_rank_deficient" in r.flags
        assert set(r.residuals) == set(a.residuals) == {"N", "E", "S"}
        assert torch.allclose(r.residuals["N"][:1], a.residuals["N"][:1], rtol=1e-6)


def test_structured_step_per_sample_flags_do_not_depend_on_the_batch():
    from p3b.latent.arms_structured import structured_step
    from test_arm_s import D, K, _module, _rand
    s, raw = _module()
    torch.manual_seed(8)
    z, feat = 0.1 * _rand(4, D), _rand(4, K)
    h = torch.full((4, 1), 0.5, dtype=F64)
    with torch.no_grad():
        zb, fb = structured_step(s, z, feat, raw, h)
        for r in range(4):
            z1, f1 = structured_step(s, z[r:r + 1], feat[r:r + 1], raw[r:r + 1], h[r:r + 1])
            assert torch.allclose(zb[r], z1[0], rtol=1e-11, atol=1e-14)
            for k, v in fb.items():
                if v.dtype == torch.bool:
                    assert bool(v[r]) == bool(f1[k][0]), k


def test_rollouts_batches_a_whole_set_in_chunks_and_the_pilot_uses_it(ds, monkeypatch):
    from p3b.eval import rollout as ro
    from p3b.pilot import runs
    std = Standardiser.fit(ds.X, NC, 3, 2, features=ds.features)
    arm = Arm("U", Smooth(ds.X.shape[-1], ds.features.shape[-1]), Identity(), std, None)
    chunks = []
    real = ro.rollout_batch

    def spy(arm_, ds_, js, n, k, track=False):
        chunks.append(list(js))
        return real(arm_, ds_, js, n, k, track)
    monkeypatch.setattr(ro, "rollout_batch", spy)
    rs = ro.rollouts(arm, ds, n_cycles=1, h_steps=1, batch=2)
    assert chunks == [[0, 1], [2]] and len(rs) == 3
    monkeypatch.setattr(ro, "rollout", lambda *a, **k: (_ for _ in ()).throw(AssertionError("unbatched")))
    chunks.clear()
    loss = runs.validation_loss(arm, ds, h_steps=1)
    assert loss == loss and chunks == [[0, 1, 2]]
