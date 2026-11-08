"""Plan 2, Task 2: data windows and standardisation from Plan 1 npz files."""
import numpy as np
import pytest
import torch

from p3b.generate import generate_trajectory
from p3b.latent.data import Standardiser, load_set, windows

POINT_A = dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=0, n_cycles=2)
POINT_B = dict(T_hot=353.15, T_cold=298.15, v=2.0e-3, alpha=0.7, beta=0.8, index=0, n_cycles=2)
NC, NS = 8, 40


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    out = {}
    for sys_, pt, mat in (("A", POINT_A, "MOF-303"), ("B", POINT_B, None)):
        p = d / f"{sys_}.npz"
        assert generate_trajectory(sys_, pt, mat, N_z=32, N_c=NC, path=p, n_snap=NS).ok
        out[sys_] = p
    return out


def test_shapes(files):
    a = load_set([files["A"]])
    assert a.X.shape == (1, 2, NS, NC * 3 + 2) and a.X.dtype == torch.float64
    assert a.raw_inputs.shape == (1, 2, NS, 4) and a.features.shape == (1, 2, NS, 4)
    b = load_set([files["B"]])
    assert b.X.shape == (1, 2, NS, NC * 2 + 1)
    assert b.raw_inputs.shape == (1, 2, NS, 2) and b.features.shape == (1, 2, NS, 4)


def test_inputs_follow_the_schedule(files):
    a = load_set([files["A"]])
    sched = a.meta[0]["schedule"]
    half = NS // 2
    np.testing.assert_allclose(a.raw_inputs[0, 1, :half, :3].numpy(), np.tile(sched[0][2], (half, 1)))
    np.testing.assert_allclose(a.raw_inputs[0, 1, half:, :3].numpy(), np.tile(sched[1][2], (half, 1)))
    assert torch.all(a.raw_inputs[..., 3] == POINT_A["v"])
    b = load_set([files["B"]])
    assert torch.all(b.raw_inputs[0, :, :half, 0] > 0) and torch.all(b.raw_inputs[0, :, half:, 0] < 0)
    assert torch.all(b.features[0, :, :half, 3] == 1) and torch.all(b.features[0, :, half:, 3] == -1)


def test_channel_layout_matches_the_store(files):
    a = load_set([files["A"]])
    d = np.load(files["A"])
    np.testing.assert_array_equal(a.X[0, 1, 5, :NC * 3].numpy(), d["channels"][1, 5].reshape(-1).astype(float))
    np.testing.assert_array_equal(a.X[0, 1, 5, NC * 3:].numpy(), d["outlet"][1, 5].astype(float))


def test_standardiser_round_trip_and_per_type_statistics(files):
    a = load_set([files["A"]])
    s = Standardiser.fit(a.X, n_cells=NC, n_ch=3, n_out=2)
    Z = s.encode(a.X)
    assert torch.allclose(s.decode(Z), a.X, rtol=0, atol=1e-12)
    # one mean per channel type: every cell of a channel shares it
    assert torch.unique(s.mean[:NC * 3].reshape(NC, 3)[:, 0]).numel() == 1
    flat = Z[..., :NC * 3].reshape(-1, NC, 3)
    assert torch.allclose(flat.mean(dim=(0, 1)), torch.zeros(3, dtype=torch.float64), atol=1e-10)


def test_windows_cross_the_half_cycle_switch(files):
    a = load_set([files["A"]])
    rng = np.random.default_rng(0)
    x0, X, F, R_, dt = windows(a, n_windows=8, n_steps=20, rng=rng, starts=[(0, NS // 2 - 5)] * 8)
    assert x0.shape == (8, NC * 3 + 2) and X.shape == (8, 20, NC * 3 + 2)
    assert F.shape == (8, 20, 4) and R_.shape == (8, 20, 4) and dt.shape == (8, 20)
    assert torch.all(dt > 0)
    assert R_[0, 4, 1] != R_[0, 5, 1]           # T_in switches from T_ads to T_des inside the window
    # targets are the next 20 snapshots of the flattened trajectory
    flat = a.X[0].reshape(-1, a.X.shape[-1])
    s = NS // 2 - 5
    assert torch.equal(x0[0], flat[s]) and torch.equal(X[0], flat[s + 1:s + 21])


def test_windows_cross_cycle_boundaries_and_stay_inside(files):
    a = load_set([files["A"]])
    rng = np.random.default_rng(1)
    for _ in range(20):
        x0, X, F, R_, dt = windows(a, n_windows=8, n_steps=20, rng=rng)
        assert torch.isfinite(X).all() and torch.all(dt > 0)


def test_feature_standardisation_uses_training_statistics(files):
    a = load_set([files["A"]])
    s = Standardiser.fit(a.X, n_cells=NC, n_ch=3, n_out=2, features=a.features)
    G = s.encode_features(a.features)
    assert G.shape == a.features.shape and torch.isfinite(G).all()
    # a constant feature (v within one trajectory) is centred, not divided by zero
    assert torch.allclose(G[..., 3], torch.zeros_like(G[..., 3]))
