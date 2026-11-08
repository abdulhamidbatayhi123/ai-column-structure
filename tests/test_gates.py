"""Reference gates on stored trajectories (plan 1, Task 7). System A, N_z = 128, N_c = 32, 2 cycles."""
import json
import subprocess
import sys

import numpy as np
import pytest

from p3b import gates
from p3b.generate import generate_trajectory

POINT_A = dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=0, n_cycles=2)
POINT_B = dict(T_hot=353.15, T_cold=298.15, v=2.0e-3, alpha=0.7, beta=0.8, index=0, n_cycles=2)


@pytest.fixture(scope="module")
def npz_a(tmp_path_factory):
    path = tmp_path_factory.mktemp("gate") / "A_0.npz"
    res = generate_trajectory("A", POINT_A, "MOF-303", N_z=128, N_c=32, path=path)
    assert res.ok, res.message
    return path


@pytest.fixture(scope="module")
def npz_b(tmp_path_factory):
    path = tmp_path_factory.mktemp("gate") / "B_0.npz"
    res = generate_trajectory("B", POINT_B, None, N_z=128, N_c=32, path=path)
    assert res.ok, res.message
    return path


def test_meta_carries_what_the_gates_need(npz_a):
    meta = gates.load_meta(npz_a)
    assert {"commit", "system", "material", "point", "schedule", "L", "N_z", "N_c"} <= set(meta)
    assert len(meta["schedule"]) == 2


def test_stored_closure(npz_a, npz_b):
    assert gates.stored_closure(npz_a) <= 1e-5
    assert gates.stored_closure(npz_b) <= 1e-5


def test_entropy_margin_positive(npz_a, npz_b):
    assert gates.coarse_entropy_margin(npz_a, gates.model_from_meta(gates.load_meta(npz_a))) > 0
    assert gates.coarse_entropy_margin(npz_b, gates.model_from_meta(gates.load_meta(npz_b))) > 0


def test_floor_share(npz_a):
    assert gates.floor_share(npz_a) < 0.01


def test_css_cycle():
    assert gates.css_cycle([10, 6, 5.5, 5.505, 5.5051]) == 4
    assert gates.css_cycle([10, 6, 5.5]) is None


def test_release_is_the_whole_desorption_half(npz_a, npz_b):
    """W_k (PREREG Section 5) integrates the full desorption half: up to the next cycle's start,
    and up to the stored end-of-trajectory ports for the last cycle."""
    for path in (npz_a, npz_b):
        d = np.load(path)
        P = d["ports_cum"]
        net = P[..., 1] - P[..., 0]
        end = d["ports_end"][1] - d["ports_end"][0]
        W = gates.cycle_release(path)
        np.testing.assert_allclose(W, [net[1, 0] - net[0, 100], end - net[1, 100]], rtol=1e-12)


def test_css_cycle_from_npz_is_none_after_two_cycles(npz_a):
    W = gates.cycle_release(npz_a)
    assert W.shape == (2,) and np.all(W > 0)
    assert gates.css_cycle(npz_a) in (None, 2)


def test_script_prints_one_line_per_gate(npz_a):
    out = subprocess.run([sys.executable, "scripts/check_reference_gate.py", str(npz_a.parent)],
                         capture_output=True, text=True, check=True).stdout
    lines = [ln for ln in out.splitlines() if ln.startswith(("PASS", "FAIL"))]
    assert len(lines) == 7 and all(ln.startswith("PASS") for ln in lines), out
