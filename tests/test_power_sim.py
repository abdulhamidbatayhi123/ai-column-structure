"""Plan 4, Task 3: power of the full decision rules (PREREG Section 9)."""
import math

import numpy as np
import pytest

from p3b.pilot.power import crossed_values, simulate


def test_seed_effect_is_shared_across_trajectories():
    v = crossed_values(0.0, sd_traj=0.01, sd_seed=1.0, n_traj=200, n_seed=5, rng=np.random.default_rng(0),
                       seed_shared=1.0)
    # with a fully shared seed effect, columns differ by a constant: within-column SD ~ sd_traj
    assert v.std(axis=0).mean() < 0.05 and v.mean(axis=0).std() > 0.3


def test_size_at_zero_effect():
    res = simulate(0.0, sd_traj=0.3, sd_seed=0.1, reps=200, draws=300, seed=1, alpha_h=0.05)
    mc = 3 * math.sqrt(0.05 * 0.95 / 200)
    assert res["SUPPORTED"] + res["SMALL"] <= 0.05 + mc


def test_power_near_one_at_a_large_effect():
    res = simulate(math.log(0.5), sd_traj=0.2, sd_seed=0.05, reps=60, draws=300, seed=2, alpha_h=0.05 / 3)
    assert res["SUPPORTED"] > 0.95


def test_fractions_sum_to_one_and_are_reproducible():
    a = simulate(math.log(0.85), sd_traj=0.3, sd_seed=0.1, reps=40, draws=200, seed=3, alpha_h=0.05)
    b = simulate(math.log(0.85), sd_traj=0.3, sd_seed=0.1, reps=40, draws=200, seed=3, alpha_h=0.05)
    assert a == b and sum(a.values()) == pytest.approx(1.0)
