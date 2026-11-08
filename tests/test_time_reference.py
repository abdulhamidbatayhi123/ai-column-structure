"""Timing and grid-convergence harness (plan 1, Task 8): smoke test at tiny grids."""
import json
import subprocess
import sys


def run(args):
    return subprocess.run([sys.executable, "scripts/time_reference.py", *args], capture_output=True, text=True,
                          check=True)


def test_harness_runs_small(tmp_path):
    out = tmp_path / "timing.json"
    run(["--system", "A", "--material", "MOF-303", "--nz", "32,64", "--cycles", "1", "--threads", "1", "--out", str(out)])
    run(["--system", "B", "--nz", "32,64", "--cycles", "1", "--threads", "1", "--out", str(out)])
    d = json.loads(out.read_text())
    assert len(d["records"]) == 4
    for r in d["records"]:
        assert {"system", "material", "N_z", "cycle", "cpu_s", "nfev", "ok", "free_ram_gb_start", "threads"} <= set(r)
        assert r["ok"] and r["cpu_s"] > 0 and r["nfev"] > 0 and r["threads"] == 1
    assert {(r["system"], r["N_z"]) for r in d["records"]} == {("A", 32), ("A", 64), ("B", 32), ("B", 64)}
    gc = d["grid_convergence"]
    assert len(gc) == 2 and {g["system"] for g in gc} == {"A", "B"}
    a = next(g for g in gc if g["system"] == "A")
    assert (a["N_coarse"], a["N_fine"]) == (32, 64) and 0 <= a["outlet_c"] < 1 and a["value"] >= a["outlet_c"]
    assert "commit" in d


def test_corner_point_is_the_fast_corner(tmp_path):
    out = tmp_path / "t.json"
    run(["--system", "B", "--nz", "32,64", "--cycles", "1", "--point", "corner", "--out", str(out)])
    d = json.loads(out.read_text())
    p = d["records"][0]["point"]
    assert p["v"] == 3.0e-3 and p["T_hot"] == 363.0 and p["T_cold"] == 293.0
    assert d["records"][0]["point_name"] == "corner"


def test_v1_large_grid_small(tmp_path):
    out = tmp_path / "v1.json"
    subprocess.run([sys.executable, "verify/v1_large_grid.py", "--n", "40", "--out", str(out)], check=True,
                   capture_output=True, text=True)
    d = json.loads(out.read_text())
    assert d["N"] == 40 and "commit" in d
    assert d["max_rel_diff_c"] < 2e-9 and d["max_rel_diff_q"] < 1e-8 and d["max_abs_diff_T_K"] < 2e-7
