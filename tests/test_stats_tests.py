"""Plan 3, Task 6: the pre-registered statistics (PREREG Section 6)."""
import itertools

import numpy as np
import pytest
from scipy import stats as st

from p3b.stats.tests import hodges_lehmann, holm, pigeonhole_bootstrap, signed_rank, tost


def test_hodges_lehmann_is_the_median_of_walsh_averages():
    rng = np.random.default_rng(0)
    x = rng.normal(size=17)
    walsh = [(x[i] + x[j]) / 2 for i in range(17) for j in range(i, 17)]
    assert hodges_lehmann(x) == pytest.approx(np.median(walsh), rel=1e-14)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_signed_rank_matches_scipy_pratt(seed):
    rng = np.random.default_rng(seed)
    x = np.round(rng.normal(-0.2, 1.0, size=30), 1)          # rounding creates ties and zeros
    p, n0 = signed_rank(x, alternative="less")
    ref = st.wilcoxon(x, zero_method="pratt", alternative="less", method="approx", correction=False).pvalue
    assert p == pytest.approx(ref, rel=1e-10)
    assert n0 == int(np.sum(x == 0))


def test_signed_rank_exact_without_ties():
    x = np.array([-0.9, -0.5, -0.31, -0.2, 0.1, -1.4, -0.05, 0.6, -0.7, -0.33])
    p, n0 = signed_rank(x, alternative="less")
    assert p == pytest.approx(st.wilcoxon(x, alternative="less", method="exact").pvalue, rel=1e-12)
    assert n0 == 0


def test_signed_rank_all_zeros_is_not_significant():
    p, n0 = signed_rank(np.zeros(12), alternative="less")
    assert p == 1.0 and n0 == 12


def test_pigeonhole_bootstrap_resamples_trajectories_and_seeds_independently():
    rng = np.random.default_rng(4)
    # crossed design: a shared seed effect (one bad seed) plus trajectory noise
    seed_eff = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    vals = rng.normal(0, 0.1, size=(48, 5)) + seed_eff[None, :]
    draws = pigeonhole_bootstrap(vals, hodges_lehmann, draws=2000, seed=0)
    assert draws.shape == (2000,) and np.isfinite(draws).all()
    # a seeds-within-trajectory bootstrap averages the bad seed away and is narrower
    nested = []
    r = np.random.default_rng(1)
    for _ in range(2000):
        ti = r.integers(0, 48, 48)
        si = r.integers(0, 5, (48, 5))
        nested.append(hodges_lehmann(vals[ti[:, None], si].mean(1)))
    assert np.std(draws) > 2 * np.std(nested)


def test_pigeonhole_bootstrap_with_one_seed_repeated_is_finite():
    vals = np.arange(20.0).reshape(4, 5)
    d = pigeonhole_bootstrap(vals, np.mean, draws=50, seed=3)
    assert np.isfinite(d).all()


def test_tost():
    rng = np.random.default_rng(5)
    inside = rng.normal(0.0, 0.05, size=48)
    outside = rng.normal(0.3, 0.05, size=48)
    assert tost(inside, margin=np.log(1.10), alpha=0.05)["equivalent"]
    assert not tost(outside, margin=np.log(1.10), alpha=0.05)["equivalent"]


def test_holm_with_an_unreached_hypothesis():
    out = holm({"H1": 0.01, "H2": 0.04, "H4": 0.03}, alpha=0.05)
    # sorted: H1 0.01 <= 0.05/3 reject; H4 0.03 > 0.05/2 stop; H2 unreached
    assert out["H1"] == (pytest.approx(0.05 / 3), True)
    assert out["H4"] == (pytest.approx(0.05 / 2), False)
    assert out["H2"] == (pytest.approx(0.05 / 1), False)        # level if step-down had continued
    out = holm({"a": 0.001, "b": 0.002}, alpha=0.05)
    assert out["a"][1] and out["b"][1]
