"""Reference timing and grid convergence (the feasibility measurement; plan 1 Task 8).

    python scripts/time_reference.py --system A --material MOF-303 --nz 512,1024,2048 --cycles 2 --threads 1
    python scripts/time_reference.py --system B --nz 256,512,1024 --cycles 2

Runs one trajectory per grid at the centre of the system's training box (fixed below) and records,
per grid and cycle, the CPU seconds (time.process_time), the right-hand-side calls and success. For
each consecutive pair of grids (N/2, N) it records Paper 1's grid_convergence() criterion: the
largest change of the outlet curve relative to its maximum (system A: outlet concentration), and,
for the temperature, the largest change relative to the outlet temperature's swing. Records are
appended to --out (default results/timing_reference.json) with the commit hash and the free RAM.
"""
import argparse
import json
import os
import sys


def parse():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--system", choices=("A", "B"), required=True)
    ap.add_argument("--material", default="MOF-303")
    ap.add_argument("--nz", required=True, help="comma-separated grids, e.g. 512,1024,2048")
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


ARGS = parse()
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # before numpy is imported
    os.environ[var] = str(ARGS.threads)

import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from p3b.design import BOXES  # noqa: E402
from p3b.provenance import git_commit  # noqa: E402
from p3b.simulate_a import simulate_trajectory_a  # noqa: E402
from p3b.simulate_b import simulate_trajectory_b  # noqa: E402


def free_ram_gb() -> float | None:
    """Available physical memory in GB, without extra dependencies (Linux, Windows)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        ms = MS()
        ms.dwLength = ctypes.sizeof(MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            return ms.ullAvailPhys / 1024 ** 3
    except (AttributeError, OSError):
        pass
    return None


def centre_point(system: str, cycles: int) -> dict:
    box = BOXES["B" if system == "B" else "A"]
    return {**{k: 0.5 * (lo + hi) for k, (lo, hi) in box.items()}, "index": 0, "n_cycles": cycles}


def outlet_collector(system: str, N: int, store: list):
    def on_cycle(k, t, y, ports):
        if system == "A":
            store.append(y[:, [N - 1, 2 * N - 1, 3 * N - 1]].copy())     # n, a, u of the outlet cell
        else:
            direction = ports[:, 3]
            store.append(np.where(direction > 0, y[:, N - 1], y[:, 0]).copy())   # u_f at the outlet
    return on_cycle


def main():
    a = ARGS
    grids = [int(x) for x in a.nz.split(",")]
    point = centre_point(a.system, a.cycles)
    material = a.material if a.system == "A" else None
    ram = free_ram_gb()
    commit = git_commit()
    records, outlets, models = [], {}, {}
    for N in grids:
        store: list = []
        if a.system == "A":
            from p3b.simulate_a import setup_a
            models[N] = setup_a(point, material, N)[0]
            res = simulate_trajectory_a(point, material, N, on_cycle=outlet_collector("A", N, store))
        else:
            from p3b.simulate_b import make_store
            models[N] = make_store(N)
            res = simulate_trajectory_b(point, N, on_cycle=outlet_collector("B", N, store))
        for k in range(a.cycles):
            ok = res.ok or (res.failed_cycle is not None and k < res.failed_cycle)
            records.append({"system": a.system, "material": material, "N_z": N, "cycle": k + 1,
                            "cpu_s": res.cycle_cpu_s[k] if k < len(res.cycle_cpu_s) else None,
                            "nfev": res.nfev[k] if k < len(res.nfev) else None, "ok": bool(ok),
                            "message": res.message, "free_ram_gb_start": ram, "threads": a.threads,
                            "point": point, "commit": commit})
            print(json.dumps(records[-1]), flush=True)
        outlets[N] = np.concatenate(store) if store else None
    gc = []
    for Nc, Nf in zip(grids[:-1], grids[1:]):
        if outlets[Nc] is None or outlets[Nf] is None or outlets[Nc].shape != outlets[Nf].shape:
            gc.append({"system": a.system, "material": material, "N_coarse": Nc, "N_fine": Nf, "value": None})
            continue
        rec = {"system": a.system, "material": material, "N_coarse": Nc, "N_fine": Nf, "commit": commit}
        if a.system == "A":
            out = {}
            for N in (Nc, Nf):
                col, o = models[N], outlets[N]
                out[N] = (o[:, 0] / col.eps, col.temperature(o[:, 0], o[:, 1], o[:, 2]))
            cc, cf = out[Nc][0], out[Nf][0]
            Tc, Tf = out[Nc][1], out[Nf][1]
            rec["outlet_c"] = float(np.abs(cc - cf).max() / np.abs(cf).max())   # Paper 1's criterion
            rec["outlet_T_rel_swing"] = float(np.abs(Tc - Tf).max() / max(np.ptp(Tf), 1e-12))
            rec["value"] = max(rec["outlet_c"], rec["outlet_T_rel_swing"])
        else:
            from p3b.thermal_store import T0
            Tc = T0 + outlets[Nc] / models[Nc].Cf
            Tf = T0 + outlets[Nf] / models[Nf].Cf
            rec["outlet_T_rel_swing"] = float(np.abs(Tc - Tf).max() / max(np.ptp(Tf), 1e-12))
            rec["value"] = rec["outlet_T_rel_swing"]
        gc.append(rec)
        print(json.dumps(rec), flush=True)
    out_path = a.out or os.path.join(ROOT, "results", "timing_reference.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    data = {"records": [], "grid_convergence": []}
    if os.path.exists(out_path):
        with open(out_path) as f:
            data = json.load(f)
    data["records"] += records
    data["grid_convergence"] += gc
    data["commit"] = commit
    with open(out_path, "w") as f:
        json.dump(data, f, indent=1)


if __name__ == "__main__":
    main()
