"""Plan 3, Task 3: one trainer for the learned arms (PREREG Section 4)."""
import copy

import pytest
import torch

from p3b.generate import generate_trajectory
from p3b.latent.arms_structured import SDynamics
from p3b.latent.arms_unstructured import UDynamics, UFDynamics
from p3b.latent.autoencoder import MLPAutoencoder, train_autoencoder
from p3b.latent.data import Standardiser, load_set
from p3b.latent.fno import FNO1d
from p3b.latent.functionals import LatentFunctionals
from p3b.latent.integrate import stage_quadrature
from p3b.latent.physics import ColumnPhysics
from p3b.simulate_a import setup_a
from p3b.train.trainer import Arm, TermsStandardiser, TrainConfig, time_scale, train

F64 = torch.float64
NC, NS, D = 8, 40, 6
POINT_A = dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=0, n_cycles=2)
POINT_A2 = dict(RH_ads=0.30, T_ads=298.15, T_des=343.15, v=0.15, alpha=0.9, beta=0.7, index=1, n_cycles=2)


def test_stage_quadrature_with_a_per_sample_step():
    h = torch.tensor([[1.0], [2.0]], dtype=F64)
    q = stage_quadrature([torch.ones(2, dtype=F64)] * 4, h)
    assert q.shape == (2,) and q.tolist() == [1.0, 2.0]


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("tr")
    paths = []
    for i, pt in enumerate((POINT_A, POINT_A2)):
        p = d / f"A{i}.npz"
        assert generate_trajectory("A", pt, "MOF-303", N_z=32, N_c=NC, path=p, n_snap=NS).ok
        paths.append(p)
    ds = load_set(paths)
    std = Standardiser.fit(ds.X, NC, 3, 2, features=ds.features)
    torch.manual_seed(0)
    ae = MLPAutoencoder(NC * 3 + 2, D, width=32)
    train_autoencoder(ae, std.encode(ds.X), epochs=30, batch=64, seed=0)
    lf = LatentFunctionals(ae.decode, std, ColumnPhysics(setup_a(POINT_A, "MOF-303", NC)[0]), "A", NC)
    return ds, std, ae, lf, time_scale(ds)


def _fixed_loss(arm, ds, cfg_kw, seed=123, n_windows=8):
    """The rollout loss on the batch of windows that seed `seed` draws first, without training."""
    import numpy as np
    from p3b.latent.data import windows
    from p3b.train.trainer import _window_loss
    cfg = TrainConfig(lr=0.0, iterations=1, seed=0, **cfg_kw)
    x0, X, F, R, dt = windows(ds, n_windows, cfg.n_steps, np.random.default_rng(seed))
    with torch.no_grad():
        return float(_window_loss(arm, cfg, x0, arm.std.encode_features(F), R, dt, arm.std.encode(X),
                                  {"flags": {k: 0 for k in ("rank_deficient", "kappa_binds",
                                                            "projection_rank_deficient", "projection_converged")}},
                                  None, arm.std))


def _decreases(hist, k=5):
    L = [x for x in hist["loss"] if x == x]                 # skipped iterations are NaN
    return len(L) >= 2 * k and sum(L[-k:]) / k < sum(L[:k]) / k


def test_u_trains(setup):
    ds, std, ae, lf, ts_ = setup
    torch.manual_seed(1)
    arm = Arm("U", UDynamics(D, 4, 16, time_scale=ts_), ae, std, lf)
    before = _fixed_loss(arm, ds, dict(n_steps=5), seed=1, n_windows=4)
    h = train(arm, ds, TrainConfig(lr=1e-4, iterations=1, seed=1, n_windows=4, n_steps=5, lr_min=1e-4))
    assert h["skipped"] == 0 and h["params"] > 0 and h["wall_s"] > 0
    assert h["loss"][0] == pytest.approx(before, rel=1e-12)                # the batch the trainer drew
    assert _fixed_loss(arm, ds, dict(n_steps=5), seed=1, n_windows=4) < before   # one step descends


def test_same_seed_same_weights(setup):
    ds, std, ae, lf, ts_ = setup
    out = []
    for _ in range(2):
        torch.manual_seed(2)
        arm = Arm("U", UDynamics(D, 4, 16, time_scale=ts_), ae, std, lf)
        train(arm, ds, TrainConfig(lr=1e-3, iterations=5, seed=7, n_windows=2, n_steps=3))
        out.append(copy.deepcopy(arm.dyn.state_dict()))
    assert all(torch.equal(out[0][k], out[1][k]) for k in out[0])


def test_cosine_schedule_ends_at_lr_min(setup):
    ds, std, ae, lf, ts_ = setup
    arm = Arm("U", UDynamics(D, 4, 16, time_scale=ts_), ae, std, lf)
    h = train(arm, ds, TrainConfig(lr=1e-3, iterations=10, seed=0, n_windows=2, n_steps=2))
    assert h["lr"][0] == pytest.approx(1e-3) and h["lr"][-1] == pytest.approx(1e-5, rel=1e-6)


def test_s_trains_and_the_autoencoder_stays_frozen(setup):
    ds, std, ae, lf, ts_ = setup
    before = copy.deepcopy(ae.state_dict())
    torch.manual_seed(3)
    # kappa damped far above the PREREG value: this toy autoencoder's entropy gradient is not
    # consistent with its energy and water gradients, so the undamped correction is stiff (the
    # pilot's h rule and H3a measure that on real decoders); here only the trainer is tested
    arm = Arm("S", SDynamics(D, 4, lf, kappa=1e4, width=16, time_scale=ts_), ae, std, lf)
    loss0 = _fixed_loss(arm, ds, dict(n_steps=3, h_steps=5), seed=3, n_windows=2)
    h = train(arm, ds, TrainConfig(lr=1e-4, iterations=1, seed=3, n_windows=2, n_steps=3, h_steps=5, lr_min=1e-4))
    assert h["skipped"] == 0 and h["flags"]["projection_converged"] > 0
    assert _fixed_loss(arm, ds, dict(n_steps=3, h_steps=5), seed=3, n_windows=2) < loss0
    assert all(torch.equal(before[k], v) for k, v in ae.state_dict().items())
    assert {"rank_deficient", "kappa_binds", "projection_converged"} <= set(h["flags"])


def test_uf_uses_standardised_terms(setup):
    ds, std, ae, lf, ts_ = setup
    ts = TermsStandardiser.fit(lf, ae, std, ds)
    z = ae.encode(std.encode(ds.X[0, 0, :5]))
    t = ts.transform(lf.evaluate(z, ds.raw_inputs[0, 0, :5], grads=True))
    for k in ("E", "N", "S", "P_E", "P_N", "Phi_S"):
        assert float(t[k].abs().max()) < 50
    arm = Arm("U+F", UFDynamics(D, 4, 16, "A", time_scale=ts_), ae, std, lf, terms=ts)
    h = train(arm, ds, TrainConfig(lr=3e-3, iterations=8, seed=4, n_windows=2, n_steps=3))
    assert len(h["loss"]) == 8


def test_penalty_weight_is_one_percent_of_the_initial_data_loss(setup):
    ds, std, ae, lf, ts_ = setup
    torch.manual_seed(5)
    arm = Arm("P", UDynamics(D, 4, 16, time_scale=ts_), ae, std, lf)
    h = train(arm, ds, TrainConfig(lr=1e-3, iterations=3, seed=5, n_windows=2, n_steps=3))
    assert h["penalty_weight"] * h["penalty0"] == pytest.approx(0.01 * h["data_loss0"], rel=1e-9)


def test_fno_trains(setup):
    ds, std, ae, lf, ts_ = setup
    torch.manual_seed(6)
    arm = Arm("FNO", FNO1d(NC, 3, 2, 4, modes=4, width=8), None, std, None)
    before = _fixed_loss(arm, ds, dict(n_steps=4), seed=6, n_windows=4)
    h = train(arm, ds, TrainConfig(lr=1e-4, iterations=1, seed=6, n_windows=4, n_steps=4, lr_min=1e-4))
    assert h["skipped"] == 0 and _fixed_loss(arm, ds, dict(n_steps=4), seed=6, n_windows=4) < before


def test_time_scale_is_the_median_snapshot_interval(setup):
    ds, *_ , ts_ = setup
    assert ts_ == pytest.approx(float(torch.median(ds.t[..., 1:] - ds.t[..., :-1])))


def test_a_diverging_window_is_skipped_not_applied(setup):
    ds, std, ae, lf, ts_ = setup
    torch.manual_seed(8)
    dyn = UDynamics(D, 4, 16, time_scale=ts_ * 1e-6)          # absurdly fast: rollouts overflow
    arm = Arm("U", dyn, ae, std, lf)
    before = copy.deepcopy(dyn.state_dict())
    h = train(arm, ds, TrainConfig(lr=1e-3, iterations=3, seed=8, n_windows=2, n_steps=20))
    if h["skipped"] == 3:
        assert all(torch.equal(before[k], v) for k, v in dyn.state_dict().items())
    assert h["skipped"] + sum(1 for x in h["loss"] if x == x) == 3
