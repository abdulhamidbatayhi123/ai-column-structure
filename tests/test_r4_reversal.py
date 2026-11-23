"""Pre-freeze revision R4 (2026-11-23): flow-reversal intervals and the floor of tau_S.

System B reverses its flow at every half-cycle, and the stored (and decoded) outlet channel is the temperature at
the current outflow end, so it changes end across the reversal interval. A residual over that interval mixes the
old direction's inputs with the new end's outlet: on the R3 pilot data it made every negative entropy residual of B
and inflated the 99.9th percentile of B's energy residual about 900-fold (255,000 against 280.5 elsewhere). The
reversal intervals are therefore left out of every residual-based statistic (reference floors, tau_S, every arm's
violation counts), and their number is reported. And tau_S gets the floor tau_ref already has (10 x rtol x the median
entropy turnover rate), so it is never exactly 0 (MOF-303's validation data had no negative production at all)."""
import numpy as np
import pytest
import torch

from p3b.generate import generate_trajectory


def _raw_b(n_cycles=2, ns=10):
    half = ns // 2
    v = np.r_[np.full(half, 2e-3), np.full(half, -2e-3)]
    raw = np.stack([v, np.full(ns, 353.15)], axis=-1)
    return torch.tensor(np.tile(raw, (n_cycles, 1)))


def test_reversal_intervals_are_where_the_flow_changes_sign():
    from p3b.eval.violations import reversal_mask
    m = reversal_mask(_raw_b(), "B")
    assert m.shape == (19,) and m.dtype == torch.bool
    assert torch.nonzero(m).flatten().tolist() == [4, 9, 14]
    raw_a = torch.ones(20, 4)
    assert not reversal_mask(raw_a, "A").any()


def test_violation_counts_leave_out_reversal_intervals_and_report_them():
    from p3b.eval.violations import violation_counts
    n = 19
    res = {"E": torch.zeros(n, dtype=torch.float64), "S": torch.ones(n, dtype=torch.float64)}
    res["E"][9] = 1e6                       # the reversal-interval artifact
    res["S"][9] = -1e6
    res["E"][3] = 1e6                       # a real violation elsewhere
    mask = torch.zeros(n, dtype=torch.bool)
    mask[[4, 9, 14]] = True
    out = violation_counts(res, {"energy_scale": torch.ones(n, dtype=torch.float64)}, {"E": 1.0}, tau_S=0.5,
                           dt=torch.ones(n, dtype=torch.float64), mask=mask)
    assert out["violations"] == {"N": None, "E": 1, "S": 0}
    assert out["excluded_intervals"] == 3


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


def test_reference_noise_leaves_out_reversal_intervals_and_gives_the_floor(b_paths):
    from p3b.pilot.runs import reference_noise
    rn = reference_noise(b_paths)
    n_snap, n_cycles = 20, 2
    per_traj = n_snap * n_cycles - 1
    n_rev = 2 * n_cycles - 1                    # reversals inside the stored record (the last one is past its end)
    assert len(rn["ref_production"]) == 2 * (per_traj - n_rev)
    assert rn["tau_floor"] > 0 and rn["excluded_intervals"] == 2 * n_rev


def test_pilot_revision_is_r4():
    from p3b.pilot.pieces import PILOT_REVISION
    assert PILOT_REVISION == "r4"
