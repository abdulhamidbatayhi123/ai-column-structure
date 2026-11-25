"""Pre-freeze revision R4 (2026-11-23), second part: tau_S gets the floor tau_ref already has (10 x rtol x the
median entropy turnover rate), so it is never exactly 0. On the R3 pilot, MOF-303's long validation data had no
negative model-style production at all and tau_S was 0. The first part of R4 (system B's outlet carried as both end
temperatures) is tested in test_r4_outlet_ends.py; with it, no interval has to be left out of any residual."""
import pytest

from p3b.generate import generate_trajectory


def test_tau_S_has_the_floor_of_tau_ref():
    from p3b.pilot.rules import tau_S_from
    assert tau_S_from([0.3, 0.5, 1.0], floor=2e-6) == 2e-6                     # no negative production
    neg = [-0.1] * 10 + [1.0] * 990
    assert tau_S_from(neg, floor=2e-6) == pytest.approx(1.0)                    # 10 x 0.1 dominates
    with pytest.raises(TypeError):
        tau_S_from([1.0])                                                        # the floor is required


@pytest.fixture(scope="module")
def b_paths(tmp_path_factory):
    d = tmp_path_factory.mktemp("r4b")
    paths = []
    for i in range(2):
        pt = dict(T_hot=353.15, T_cold=298.15, v=2.0e-3 * (1 + 0.5 * i), alpha=0.7, beta=0.8, index=i, n_cycles=2)
        p = d / f"t{i}.npz"
        assert generate_trajectory("B", pt, None, N_z=32, N_c=8, path=p, n_snap=20).ok
        paths.append(p)
    return paths


def test_reference_noise_keeps_every_interval_and_gives_the_floor(b_paths):
    from p3b.pilot.runs import reference_noise
    rn = reference_noise(b_paths)
    assert len(rn["ref_production"]) == 2 * (20 * 2 - 1)
    assert rn["tau_floor"] > 0 and rn["tau_ref"] >= rn["tau_floor"]

