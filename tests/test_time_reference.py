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
