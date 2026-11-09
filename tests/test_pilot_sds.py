"""Plan 5 Task 5: the pilot SDs (sds.json) and the positive control (positive_control.json)."""
import math

import numpy as np
import pytest

from p3b.pilot import power, runs


def _rows(n_traj, scale, n_cycles=60):
    """Rows like p3b.eval.evaluate's: trajectory j's errors are scale[j] at every cycle."""
    return [{"trajectory": j, "e": [float(scale[j])] * n_cycles, "eps": [2.0 * float(scale[j])] * n_cycles}
            for j in range(n_traj)]


def test_hypothesis_statistics_follow_the_prereg_table():
    row = {"trajectory": 0, "e": [float(k + 1) for k in range(60)], "eps": [float(10 * (k + 1)) for k in range(60)]}
    assert runs.hypothesis_error(row, "H1") == 500.0                     # eps_W at cycle 50
    assert runs.hypothesis_error(row, "H5") == 200.0                     # eps_W at cycle 20
    assert runs.hypothesis_error(row, "H2") == pytest.approx(3.5)        # mean e over cycles 1-6
    assert runs.hypothesis_error(row, "H4") == pytest.approx(3.5)


def test_hypothesis_error_refuses_a_short_rollout():
    with pytest.raises(ValueError):
        runs.hypothesis_error({"trajectory": 0, "e": [1.0] * 10, "eps": [1.0] * 10}, "H1")


def test_sds_values_are_log_ratios_s_over_u_per_trajectory_and_seed():
    n = 4
    u = {0: _rows(n, [1.0, 2.0, 3.0, 4.0]), 1: _rows(n, [1.0, 1.0, 1.0, 1.0])}
    s = {0: _rows(n, [0.5, 1.0, 1.5, 2.0]), 1: _rows(n, [2.0, 2.0, 2.0, 2.0])}
    out = runs.sds_values(u, s)
    assert set(out) == {"H1", "H2", "H4", "H5"}
    v = np.asarray(out["H1"])
    assert v.shape == (n, 2)
    assert np.allclose(v[:, 0], math.log(0.5)) and np.allclose(v[:, 1], math.log(2.0))


def test_sds_values_need_the_same_seeds_for_both_arms():
    with pytest.raises(ValueError):
        runs.sds_values({0: _rows(2, [1, 1])}, {1: _rows(2, [1, 1])})


def test_sds_values_feed_upper80_sd():
    from p3b.pilot.rules import upper80_sd
    rng = np.random.default_rng(0)
    u = {k: _rows(16, np.exp(rng.normal(size=16))) for k in (0, 1)}
    s = {k: _rows(16, np.exp(rng.normal(size=16))) for k in (0, 1)}
    s_t, s_s = upper80_sd(runs.sds_values(u, s)["H4"])
    assert s_t > 0 and s_s > 0


def test_directional_rule_is_the_power_simulations_rule():
    rng = np.random.default_rng(1)
    strong = np.log(0.5) + 0.05 * rng.normal(size=(16, 2))
    assert power.directional_rule(strong, alpha_h=0.05, draws=500, seed=0) == "SUPPORTED"
    null = 0.3 * rng.normal(size=(16, 2))
    assert power.directional_rule(null, alpha_h=0.05, draws=500, seed=0) in ("INCONCLUSIVE", "REFUTED")


def test_positive_control_word_on_a_clear_training_effect():
    rng = np.random.default_rng(2)
    full = {k: _rows(16, 0.5 * np.exp(0.05 * rng.normal(size=16))) for k in (0, 1)}
    quarter = {k: _rows(16, np.exp(0.05 * rng.normal(size=16))) for k in (0, 1)}
    out = runs.positive_control(full, quarter, draws=500)
    assert out["word"] == "SUPPORTED" and out["hypothesis"] == "H2"
    assert set(out) == {"word", "hypothesis", "alpha"}          # blinding: no error or ratio is written


def test_wall_per_iter_refuses_a_timing_made_of_skipped_windows(monkeypatch):
    """Skipped (diverged) windows cost almost nothing, so they would make an arm look cheap and set
    U-long's iteration count too small."""
    from p3b.train import trainer

    def fake_train(arm, ds, cfg):
        return {"skipped": cfg.iterations, "skipped_grad": 0, "loss": []}
    monkeypatch.setattr(trainer, "train", fake_train)
    with pytest.raises(RuntimeError):
        runs.wall_per_iter({"S": lambda: None}, ds=None, iterations=4, warmup=0)
