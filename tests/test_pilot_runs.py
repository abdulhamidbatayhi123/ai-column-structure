"""Plan 5: the pilot runner, which writes the inputs of scripts/pilot_rules.py from the pilot data."""
import json

import numpy as np
import pytest

from p3b.generate import generate_trajectory
from p3b.pilot import runs

POINTS = [dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=i, n_cycles=3)
          for i in range(2)]


@pytest.fixture(scope="module")
def pilot_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("pilot")
    for split in ("train", "validation"):
        (d / split).mkdir()
        for pt in POINTS:
            pt = dict(pt, RH_ads=pt["RH_ads"] + 0.05 * pt["index"])
            assert generate_trajectory("A", pt, "MOF-303", N_z=32, N_c=8, path=d / split / f"t{pt['index']}.npz",
                                       n_snap=40).ok
    return d


def test_reference_noise_gives_floors_production_and_a_pilot_wide_tau_ref(pilot_dir):
    out = runs.reference_noise(sorted((pilot_dir / "validation").glob("*.npz")))
    assert set(out) == {"floors", "ref_production", "tau_ref"}
    assert {"N", "E"} <= set(out["floors"]) and all(v > 0 for v in out["floors"].values())
    assert len(out["ref_production"]) > 0 and np.isfinite(out["ref_production"]).all()
    assert out["tau_ref"] > 0


def test_css_per_trajectory_with_a_sentinel_when_not_reached(pilot_dir):
    c = runs.css_cycles(sorted((pilot_dir / "validation").glob("*.npz")))
    assert len(c) == 2 and all(isinstance(x, int) for x in c)
    assert all(2 <= x <= 4 for x in c)                       # 3 cycles stored: reached by 3, else 4


def test_decoder_gate_grid(pilot_dir):
    g = runs.decoder_gate_grid(sorted((pilot_dir / "train").glob("*.npz")),
                               sorted((pilot_dir / "validation").glob("*.npz")),
                               d_grid=(4, 8), epochs=1, width=8, seed=0)
    assert set(g) == {4, 8} and all(isinstance(v, bool) for v in g.values())


def test_write_inputs_creates_the_files_pilot_rules_reads(pilot_dir, tmp_path):
    files = runs.write_data_inputs({"A_MOF-303": pilot_dir}, tmp_path, d_grid=(4,), epochs=1, width=8)
    for name in ("css.json", "decoder_gate.json", "reference_noise.json"):
        assert (tmp_path / name).exists()
    css = json.loads((tmp_path / "css.json").read_text())
    assert set(css) == {"A_MOF-303"} and files


def _setup(pilot_dir):
    import torch
    from p3b.latent.autoencoder import MLPAutoencoder, train_autoencoder
    from p3b.latent.data import Standardiser, load_set
    from p3b.latent.functionals import LatentFunctionals
    ds = load_set(sorted((pilot_dir / "train").glob("*.npz")))
    val = load_set(sorted((pilot_dir / "validation").glob("*.npz")))
    std = Standardiser.fit(ds.X, 8, 3, 2, features=ds.features)
    torch.manual_seed(0)
    ae = MLPAutoencoder(26, 4, width=8)
    train_autoencoder(ae, std.encode(ds.X), epochs=2, seed=0)
    lf = LatentFunctionals(ae.decode, std, runs._physics(ds), "A", 8)
    return ds, val, std, ae, lf


def test_kappa_from_encoded_data_is_positive(pilot_dir):
    ds, val, std, ae, lf = _setup(pilot_dir)
    k = runs.kappa_from_data(lf, ae, std, ds)
    assert k > 0 and np.isfinite(k)


def test_train_curve_records_validation_loss_at_checkpoints(pilot_dir):
    from p3b.latent.arms_unstructured import UDynamics
    from p3b.train.trainer import Arm, time_scale
    ds, val, std, ae, lf = _setup(pilot_dir)
    arm = Arm("U", UDynamics(4, 4, 8, time_scale=time_scale(ds)), ae, std, lf)
    curve = runs.train_curve(arm, ds, val, lr=1e-3, iterations=4, every=2, h_steps=1, seed=0,
                             n_windows=1, n_steps=2)
    assert [it for it, _ in curve] == [2, 4] and all(np.isfinite(v) or v != v for _, v in curve)


def test_lr_sweep_writes_one_loss_per_rate(pilot_dir):
    from p3b.latent.arms_unstructured import UDynamics
    from p3b.train.trainer import Arm, time_scale
    ds, val, std, ae, lf = _setup(pilot_dir)

    def make():
        return Arm("U", UDynamics(4, 4, 8, time_scale=time_scale(ds)), ae, std, lf)

    out = runs.lr_sweep(make, ds, val, rates=(3e-4, 1e-3), iterations=2, h_steps=1, seed=0, n_windows=1, n_steps=2)
    assert set(out) == {3e-4, 1e-3}
