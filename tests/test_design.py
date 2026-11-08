"""Design sets (plan 1, Task 3): sizes, boxes, reproducibility, nesting, disjointness."""
import itertools

import pytest

from p3b import design as d

EXPECTED_SIZES = {
    ("A_primary", "pilot_train"): (32, 6), ("A_primary", "pilot_val"): (16, 60),
    ("A_primary", "train"): (256, 6), ("A_primary", "validation"): (32, 6),
    ("A_primary", "long_validation"): (4, 60), ("A_primary", "long_test"): (48, 60),
    ("A_primary", "ood"): (24, 30), ("A_primary", "scenarios"): (8, None),
}
for _sys in ("A_replication", "B"):
    EXPECTED_SIZES.update({(_sys, "pilot"): (8, 60), (_sys, "train"): (256, 6), (_sys, "validation"): (32, 6),
                           (_sys, "long_validation"): (4, 60), (_sys, "long_test"): (48, 60)})

SAMPLED_KEYS = {"A": {"RH_ads", "T_ads", "T_des", "v", "alpha", "beta"},
                "B": {"T_hot", "T_cold", "v", "alpha", "beta"},
                "scenarios": {"RH_ads", "T_ads", "v"}}


def test_sizes():
    assert d.SET_SIZES == EXPECTED_SIZES
    for (system, name), (n, cyc) in EXPECTED_SIZES.items():
        pts = d.make_set(system, name, seed=11)
        assert len(pts) == n
        assert all(p["n_cycles"] == cyc for p in pts)
        assert [p["index"] for p in pts] == list(range(n))


def test_box_values_are_the_global_constraints():
    assert d.BOXES["A"] == {"RH_ads": (0.20, 0.60), "T_ads": (293.0, 308.0), "T_des": (333.0, 363.0),
                            "v": (0.03, 0.30), "alpha": (0.5, 1.2), "beta": (0.5, 1.0)}
    assert d.BOXES["A_OOD"] == dict(d.BOXES["A"], RH_ads=(0.10, 0.20), T_des=(363.0, 383.0))
    assert d.BOXES["B"] == {"T_hot": (343.0, 363.0), "T_cold": (293.0, 303.0), "v": (0.5e-3, 3.0e-3),
                            "alpha": (0.4, 0.9), "beta": (0.6, 1.0)}


@pytest.mark.parametrize("system,name", list(EXPECTED_SIZES))
def test_ranges(system, name):
    box = d.BOXES["A_OOD" if name == "ood" else ("B" if system == "B" else "A")]
    keys = SAMPLED_KEYS["scenarios" if name == "scenarios" else ("B" if system == "B" else "A")]
    for p in d.make_set(system, name, seed=5):
        assert keys <= set(p)
        for k in keys:
            lo, hi = box[k]
            assert lo <= p[k] <= hi, (system, name, k, p[k])


def test_ood_lies_outside_the_training_box():
    for p in d.make_set("A_primary", "ood", seed=5):
        assert p["RH_ads"] <= 0.20 and p["T_des"] >= 363.0


def test_reproducible():
    assert d.make_set("A_primary", "train", seed=123) == d.make_set("A_primary", "train", seed=123)
    assert d.make_set("A_primary", "train", seed=123) != d.make_set("A_primary", "train", seed=124)


def test_nested():
    pool = d.make_set("A_primary", "train", seed=7)
    for small, big in [(16, 32), (32, 64), (64, 128), (128, 256)]:
        assert d.train_subset(pool, big)[:small] == d.train_subset(pool, small)
    assert d.train_subset(pool, 256) == pool
    with pytest.raises(ValueError):
        d.train_subset(pool, 257)


def test_unknown_set_raises():
    with pytest.raises(KeyError):
        d.make_set("B", "ood", seed=1)


def test_sets_disjoint():
    from scripts.design_sets import SEEDS
    assert set(SEEDS) == set(EXPECTED_SIZES)
    assert len(set(SEEDS.values())) == len(SEEDS)
    for system in ("A_primary", "A_replication", "B"):
        sets = {name: d.make_set(system, name, SEEDS[(system, name)]) for (s, name) in SEEDS if s == system}
        pts = {name: {tuple(sorted((k, v) for k, v in p.items() if k not in ("index", "n_cycles"))) for p in ps}
               for name, ps in sets.items()}
        for a, b in itertools.combinations(pts, 2):
            assert not (pts[a] & pts[b]), (system, a, b)
