"""Plan 4, Task 1: the pilot's rule functions (PREREG Section 10), one test per rule."""
import numpy as np
import pytest
from scipy import stats as st

from p3b.pilot.rules import (choose_h, choose_iterations, choose_K, choose_lr, h4_margin, thresholds,
                             upper80_sd)


def test_lr_is_the_best_validation_loss_ties_to_the_smaller():
    out = choose_lr({"U": {3e-4: 0.5, 1e-3: 0.4, 3e-3: 0.6}, "S": {3e-4: 0.3, 1e-3: 0.3, 3e-3: 0.9}})
    assert out == {"U": 1e-3, "S": 3e-4}


def test_lr_with_no_finished_run_is_none():
    assert choose_lr({"U": {3e-4: float("nan"), 1e-3: float("nan"), 3e-3: float("nan")}}) == {"U": None}


def test_iterations_first_point_where_every_arm_improves_less_than_two_percent_per_1000():
    a = [(1000, 1.0), (2000, 0.8), (3000, 0.79), (4000, 0.785)]     # < 2 % from 3000 on
    b = [(1000, 1.0), (2000, 0.9), (3000, 0.85), (4000, 0.84)]      # 0.85 -> 0.84 is 1.2 %, 0.9 -> 0.85 is 5.6 %
    assert choose_iterations({"U": a, "S": b}) == 4000
    assert choose_iterations({"U": [(1000, 1.0), (2000, 0.5)]}) is None


def test_h_is_the_largest_passing():
    T = 1000.0
    rates = {T / 200: 3e-4, T / 400: 5e-5, T / 800: 1e-5}
    stiff = {T / 200: True, T / 400: True, T / 800: True}
    assert choose_h(rates, stiff) == T / 400
    assert choose_h(rates, {T / 200: True, T / 400: False, T / 800: True}) == T / 800
    assert choose_h({T / 200: 1.0, T / 400: 1.0, T / 800: 1.0}, stiff) is None


def test_thresholds():
    rng = np.random.default_rng(0)
    g2 = rng.lognormal(0, 1, 1000)
    defects = rng.poisson(2.0, 200) / 1e4                 # defect steps per step, per trajectory
    th = thresholds(floors={"N": 1e-6, "E": 2e-6}, neg_production=rng.exponential(1e-4, 500), g2=g2,
                    defects_per_step=defects, tau_ref=1e-3)
    assert th["violation_N"] == pytest.approx(1e-5) and th["violation_E"] == pytest.approx(2e-5)
    assert th["kappa"] == pytest.approx(1e-3 * np.median(g2))
    assert th["defect_bound"] <= 1e-3
    assert th["tau_ref"] == 1e-3 and th["tau_S"] > 0
    capped = thresholds(floors={"N": 1, "E": 1}, neg_production=np.ones(5), g2=g2,
                        defects_per_step=np.full(10, 0.01), tau_ref=1.0)
    assert capped["defect_bound"] == pytest.approx(1e-3)


def test_K_is_the_smallest_within_one_percent():
    assert choose_K({1: 0.05, 2: 0.012, 4: 0.006, 8: 0.003}) == 4
    assert choose_K({1: 0.05, 2: 0.03, 4: 0.02, 8: 0.015}) is None


def test_upper80_sd_on_a_chi_square_example():
    rng = np.random.default_rng(1)
    v = rng.normal(0, 1, size=(48, 5))
    sd_t, sd_s = upper80_sd(v)
    n = 48
    s = v.mean(1).std(ddof=1)
    assert sd_t == pytest.approx(s * np.sqrt((n - 1) / st.chi2.ppf(0.2, n - 1)), rel=1e-12)
    assert sd_t > s and sd_s > 0


def test_h4_margin_rule():
    assert h4_margin({1.10: 0.85, 1.15: 0.95}) == 1.10
    assert h4_margin({1.10: 0.6, 1.15: 0.81, 1.20: 0.9}) == 1.15
    assert h4_margin({1.10: 0.3, 1.15: 0.5, 1.20: 0.7}) == "demoted"
