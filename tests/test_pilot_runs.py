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


def test_spectral_radius_by_power_iteration_on_a_linear_field():
    import torch
    A = torch.diag(torch.tensor([-1.0, -5.0, -0.5], dtype=torch.float64))
    rho = runs.spectral_radius(lambda z: z @ A.T, torch.zeros(1, 3, dtype=torch.float64), iters=50)
    assert abs(rho - 5.0) < 1e-6


def test_h_check_reports_rates_stability_and_defects(pilot_dir):
    from p3b.latent.arms_structured import SDynamics
    from p3b.train.trainer import Arm, time_scale
    ds, val, std, ae, lf = _setup(pilot_dir)
    s = SDynamics(4, 4, lf, kappa=runs.kappa_from_data(lf, ae, std, ds), width=8, time_scale=time_scale(ds))
    out = runs.h_check(Arm("S", s, ae, std, lf), val, tau_S=1e-9, steps=(1, 2))
    assert set(out["h"]) == {1, 2}
    for v in out["h"].values():
        assert 0 <= v["viol_rate"] <= 1 and isinstance(v["stiff_ok"], bool) and v["h_rho"] >= 0
    assert set(out["defects_per_step"]) == {1, 2}
    assert all(len(v) == val.X.shape[0] for v in out["defects_per_step"].values())


# ---- Task 4: C+L's substeps K, wall-clock per iteration ----------------------------------------------
@pytest.fixture(scope="module")
def pilot16(tmp_path_factory):
    """One trajectory stored on 16 cells: C+L's Newton does not converge on 4 or 8 cells at these
    large snapshot intervals (a 508 s step at 4 cells), and the study runs it on 32."""
    d = tmp_path_factory.mktemp("pilot16")
    pt = dict(POINTS[0], n_cycles=2)
    assert generate_trajectory("A", pt, "MOF-303", N_z=64, N_c=16, path=d / "t0.npz", n_snap=40).ok
    return d


def test_cplusl_K_gives_one_finite_difference_per_K(pilot16):
    out = runs.cplusl_K(sorted(pilot16.glob("*.npz")), "MOF-303", n_coarse=16, Ks=(1, 2))
    assert set(out) == {1, 2}
    assert all(np.isfinite(v) and v >= 0 for v in out.values())


def test_cplusl_K_counts_a_failed_run_as_infinite(pilot_dir, monkeypatch):
    from p3b.train import cplusl

    def fail(*a, **k):
        raise RuntimeError("Newton failed")
    monkeypatch.setattr(cplusl, "rollout_cplusl_to_file", fail)
    out = runs.cplusl_K(sorted((pilot_dir / "validation").glob("*.npz")), "MOF-303", n_coarse=4, Ks=(1,))
    assert out == {1: float("inf")}


def test_cplusl_K_feeds_choose_K():
    from p3b.pilot.rules import choose_K
    assert choose_K({1: 0.02, 2: 0.004, 4: 0.001, 8: 0.0}) == 2


def test_wall_per_iter_measures_each_arm(pilot_dir):
    from p3b.latent.arms_unstructured import UDynamics
    from p3b.train.trainer import Arm, time_scale
    ds, val, std, ae, lf = _setup(pilot_dir)
    # plumbing only: S's windows diverge on this 4-d, 40-snapshot data and its timing is then refused
    # (test_wall_per_iter_refuses_a_timing_made_of_skipped_windows); the dry run times a working S
    makers = {"U": lambda: Arm("U", UDynamics(4, 4, 8, time_scale=time_scale(ds)), ae, std, lf),
              "U16": lambda: Arm("U", UDynamics(4, 4, 16, time_scale=time_scale(ds)), ae, std, lf)}
    out = runs.wall_per_iter(makers, ds, iterations=2, warmup=1, h_steps=1, n_windows=1, n_steps=2)
    assert set(out) == {"U", "U16"} and all(v > 0 and np.isfinite(v) for v in out.values())


def test_two_seed_rows_trains_a_fresh_arm_per_seed_and_evaluates_validation(pilot_dir):
    from p3b.latent.arms_unstructured import UDynamics
    from p3b.train.trainer import Arm, time_scale
    ds, val, std, ae, lf = _setup(pilot_dir)
    out, train = runs.two_seed_rows(lambda: Arm("U", UDynamics(4, 4, 8, time_scale=time_scale(ds)), ae, std, lf),
                                    ds, val, lr=1e-3, iterations=2, h_steps=1, seeds=(0, 1), n_windows=1, n_steps=2)
    assert set(out) == {0, 1} and set(train) == {0, 1} and all(t["iterations"] == 2 for t in train.values())
    for s, rows in out.items():
        assert len(rows) == val.X.shape[0] and all(r["seed"] == s for r in rows)
        assert all(len(r["e"]) == val.X.shape[1] for r in rows)


def test_cplusl_validation_loss_is_deterministic_and_finite(pilot16):
    from p3b.latent.data import load_set
    from p3b.solvers.c_plus_l import Multipliers
    from p3b.train.cplusl import cplusl_val_loss
    ds = load_set(sorted(pilot16.glob("*.npz")))
    mult = Multipliers(n_out=3, constant=1.0)
    a = cplusl_val_loss(mult, ds, "MOF-303", n_coarse=16, K=1)
    b = cplusl_val_loss(mult, ds, "MOF-303", n_coarse=16, K=1)
    assert a == b and np.isfinite(a) and a > 0


def test_cplusl_curve_and_lr_sweep(pilot16):
    from p3b.latent.data import load_set
    ds = load_set(sorted(pilot16.glob("*.npz")))
    curve = runs.cplusl_curve(ds, ds, "MOF-303", lr=1e-3, iterations=4, every=2, K=1, seed=0, n_coarse=16,
                              n_windows=1, n_steps=2)
    assert [it for it, _ in curve] == [2, 4] and all(np.isfinite(v) for _, v in curve)
    sweep = runs.cplusl_lr_sweep(ds, ds, "MOF-303", K=1, rates=(1e-3,), iterations=2, n_coarse=16, n_windows=1,
                                 n_steps=2)
    assert set(sweep) == {1e-3}


def test_g2_values_are_raw_and_kappa_is_their_scaled_median(pilot_dir):
    ds, val, std, ae, lf = _setup(pilot_dir)
    g2 = runs.g2_from_data(lf, ae, std, ds, n=16)
    assert len(g2) == 16 and all(v >= 0 for v in g2)
    assert runs.kappa_from_data(lf, ae, std, ds, n=16) == pytest.approx(1e-3 * float(np.median(g2)))


def test_h_check_makes_an_h_with_a_failed_rollout_inadmissible(pilot_dir, monkeypatch):
    """A rollout that diverges at step 1 leaves almost no residuals; its violation rate must not read 0,
    or the h rule could choose an h at which S fails (found in the Task 6 dry run)."""
    import torch
    from p3b.eval import rollout as ro
    from p3b.latent.arms_structured import SDynamics
    from p3b.pilot.rules import choose_h_steps
    from p3b.train.trainer import Arm, time_scale
    ds, val, std, ae, lf = _setup(pilot_dir)
    s = SDynamics(4, 4, lf, kappa=runs.kappa_from_data(lf, ae, std, ds), width=8, time_scale=time_scale(ds))

    def failed(arm, ds_, j, n_cycles, h_steps, track=False):
        return ro.RolloutResult(X=torch.full((1,), float("nan")), X_end=torch.full((1,), float("nan")),
                                finite=torch.tensor([False]), failed_cycle=1, cpu_s=0.0,
                                residuals={"S": torch.zeros(1)}, error="LinAlgError")
    monkeypatch.setattr(ro, "rollout", failed)
    out = runs.h_check(Arm("S", s, ae, std, lf), val, tau_S=1e-9, steps=(1, 2))
    assert all(v["viol_rate"] == 1.0 and v["failed"] == val.X.shape[0] for v in out["h"].values())
    assert choose_h_steps({k: v["viol_rate"] for k, v in out["h"].items()},
                          {k: v["stiff_ok"] for k, v in out["h"].items()}) is None
