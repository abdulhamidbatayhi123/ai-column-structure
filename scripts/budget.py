"""The training budget from measured per-iteration costs (docs/compute_3B.md, Sections 2 and 3).

Training runs per arm come from the PREREG design table and the descriptive sensitivities; the cost of
one run is seconds per iteration (one thread, `results/timing_arms.json`, Kaggle) times the iteration
count, both chosen in the pilot (PREREG 10: one iteration count for every arm, h RK4 steps per
snapshot interval). h = 4 was not timed: it is extrapolated linearly through h = 1 and 2 (cost = a + b h).
U-long trains for S's wall-clock. The reduction order is PREREG Section 11 / compute_3B Section 3.

`need` (T8 DOC-2, DOC-7, R1-6) is the whole confirmatory need, every block from a measured cost: latent
training, C+L training at the pilot's K (results/timing_cplusl_train.json; only measured K are accepted),
every autoencoder (study, gate wave, sensitivities, under the R1 loss or plain MSE), the evaluation of
every cell on every set, arm C and MUSCL-C, the H6 searches with their reference checks, and the
confirmatory reference data; in one-run CPU hours and in Kaggle session hours, the ceiling's unit (PREREG
Section 11; scripts/ceiling_ledger.py counts it). The unmeasured costs come from one timing kernel
(scripts/time_ceiling.py, results/timing_ceiling.json).

    python scripts/budget.py [--iterations 8000] [--n-css 13]
"""
from __future__ import annotations

import argparse
import json
import math
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# seeds per design cell (PREREG design table); learning curve = 3 sizes (16, 64, 128) x 3 seeds
DESIGN = {
    "U":      {"A_full": 5, "A_32": 5, "A_lc": 9, "Arep": 6, "B": 10},
    "U+F":    {"A_full": 5, "A_32": 5, "Arep": 6, "B": 10},
    "U-long": {"A_full": 3},
    "P":      {"A_full": 5, "A_32": 5, "A_lc": 9, "Arep": 6},
    "S":      {"A_full": 5, "A_32": 5, "A_lc": 9, "Arep": 6, "B": 10},
    "S-L":    {"A_full": 5, "A_32": 5, "B": 6},
    "S-M":    {"A_full": 5},
    "S-Pi":   {"A_full": 5},
    "FNO":    {"A_full": 3, "A_32": 3},
    "C+L":    {"A_full": 3, "A_32": 3},
}
# descriptive sensitivities (PREREG 6): U, U+F, S, 3 seeds each; the autoencoder loss (R1): U and S, 3 seeds
SENSITIVITIES = {"degraded": {"U": 3, "U+F": 3, "S": 3}, "latent_dim": {"U": 3, "U+F": 3, "S": 3},
                 "ae_loss": {"U": 3, "S": 3}}


def training_runs() -> dict:
    runs = {a: sum(c.values()) for a, c in DESIGN.items()}
    for sens in SENSITIVITIES.values():
        for a, n in sens.items():
            runs[a] += n
    return runs


def _s_per_iteration(timing: dict, arm: str, h: int) -> float:
    t = timing["train"]
    s1, s2 = t[f"{arm}|h1|t1"]["wall_s_per_iteration"], t[f"{arm}|h2|t1"]["wall_s_per_iteration"]
    return s1 + (s2 - s1) * (h - 1)                      # a + b h through (1, s1) and (2, s2)


def run_hours(timing: dict, arm: str, iterations: int, h: int, c_plus_l_hours: float | None = None) -> float:
    if arm == "C+L":
        if c_plus_l_hours is None:
            raise ValueError("C+L was not timed: pass c_plus_l_hours")
        return c_plus_l_hours
    return _s_per_iteration(timing, "S" if arm == "U-long" else arm, h) * iterations / 3600


def training_budget(timing: dict, iterations: int, h: int, c_plus_l_hours: float = 1.0) -> dict:
    runs = training_runs()
    hours = {a: run_hours(timing, a, iterations, h, c_plus_l_hours) for a in runs}
    per_arm = {a: runs[a] * hours[a] for a in runs}
    lc = DESIGN["U"]["A_lc"] // 9                         # one seed at each of sizes 16 and 128
    reductions = [
        ("degraded-data sensitivity", sum(n * hours[a] for a, n in SENSITIVITIES["degraded"].items())),
        ("U-long", per_arm["U-long"]),
        ("S-Pi", per_arm["S-Pi"]),
        ("latent-dimension sensitivity", sum(n * hours[a] for a, n in SENSITIVITIES["latent_dim"].items())),
        ("autoencoder-loss sensitivity", sum(n * hours[a] for a, n in SENSITIVITIES["ae_loss"].items())),
        ("MUSCL-C", 0.0),                                 # no training; its evaluation is in the C block
        ("learning-curve sizes 16 and 128 to 2 seeds", sum(2 * lc * hours[a] for a in ("U", "P", "S"))),
        ("System B S-L", DESIGN["S-L"]["B"] * hours["S-L"]),
    ]
    return {"iterations": iterations, "h": h, "runs": runs, "hours_per_run": hours, "per_arm": per_arm,
            "total": sum(per_arm.values()), "unmeasured": ["C+L"],
            "reductions": [{"step": s, "hours": v} for s, v in reductions]}


def evaluation_hours(s_per_interval: float, rollouts: int = 120, cycles: int = 60, snaps: int = 200) -> float:
    """compute_3B Section 2's evaluation line: about 120 projected-arm rollouts of a test set, 60 cycles of
    200 snapshot intervals, at a measured cost per interval for the whole batch."""
    return rollouts * cycles * snaps * s_per_interval / 3600


def evaluation_hours_from(timing2: dict, h: int, batched: bool = True, batch: int = 48) -> float:
    """From S's measured rollout step (time_arms --batch): one 48-trajectory batch, or 48 single rollouts."""
    st = timing2["step_S"]
    if batched:
        return evaluation_hours(st[f"h{h}|b{batch}"]["wall_s_per_interval"])
    return batch * evaluation_hours(st[f"h{h}|b1"]["wall_s_per_interval"])


def session_throughput(timing2: dict, arm: str, n: int, h: int = 2) -> float:
    """Runs per unit time of n one-thread jobs in one session, relative to one job alone."""
    c = timing2["concurrent"]
    alone = c[f"{arm}|h{h}|n1"]["wall_s_per_iteration"][0]
    walls = c[f"{arm}|h{h}|n{n}"]["wall_s_per_iteration"]
    return sum(alone / w for w in walls)



# ---- the whole need (T8 DOC-2, DOC-7, R1-6) --------------------------------------------------------------
# Cell-level design, mirroring p3b.train.cells.DESIGN (tests/test_budget_cells.py checks the counts there).
P_KEY, R_KEY, B_KEY = "A_MOF-303", "A_Al-fumarate", "B"
_LC = [(P_KEY, 16, 3), (P_KEY, 64, 3), (P_KEY, 128, 3)]
CELL_DESIGN = {
    "U":       [(P_KEY, "full", 5), (P_KEY, 32, 5), *_LC, (R_KEY, "full", 3), (R_KEY, 32, 3), (B_KEY, "full", 5), (B_KEY, 32, 5)],
    "U+F":     [(P_KEY, "full", 5), (P_KEY, 32, 5), (R_KEY, "full", 3), (R_KEY, 32, 3), (B_KEY, "full", 5), (B_KEY, 32, 5)],
    "U-long":  [(P_KEY, "full", 3)],
    "P":       [(P_KEY, "full", 5), (P_KEY, 32, 5), *_LC, (R_KEY, "full", 3), (R_KEY, 32, 3)],
    "S":       [(P_KEY, "full", 5), (P_KEY, 32, 5), *_LC, (R_KEY, "full", 3), (R_KEY, 32, 3), (B_KEY, "full", 5), (B_KEY, 32, 5)],
    "S-L":     [(P_KEY, "full", 5), (P_KEY, 32, 5), (B_KEY, "full", 3), (B_KEY, 32, 3)],
    "S-M":     [(P_KEY, "full", 5)],
    "S-Pi":    [(P_KEY, "full", 5)],
    "FNO":     [(P_KEY, "full", 3), (P_KEY, 32, 3)],
    "C+L":     [(P_KEY, "full", 3), (P_KEY, 32, 3)],
    "C":       [(P_KEY, "full", 1), (R_KEY, "full", 1), (B_KEY, "full", 1)],
    "MUSCL-C": [(P_KEY, "full", 1)],
}
# the descriptive sensitivities as cells (cells.sensitivity_cells): kind -> arms, (key, size), 3 seeds
SENS_CELLS = {"noise": (("U", "U+F", "S"), (P_KEY, 32)), "d+": (("U", "U+F", "S"), (P_KEY, "full")),
              "aemse": (("U", "S"), (P_KEY, "full"))}
LATENT = ("U", "U+F", "U-long", "P", "S", "S-L", "S-M", "S-Pi", "FNO")
WITH_AE = ("U", "U+F", "U-long", "P", "S", "S-L", "S-M", "S-Pi")           # cells.LEARNED_WITH_AE
D_GRID_N = 5                                                               # cells.D_GRID (16, 24, 32, 48, 64)
POOL = 256                                                                 # design.SET_SIZES train pool
AE_EPOCHS = 60                                                             # PREREG Section 4
N_SNAP = 200
BATCH = 48                                                                 # p3b.eval.rollout.BATCH
# (trajectories, cycles) per set (p3b.design.SET_SIZES); every cell is scored on the validation sets after
# training and on the test sets that exist for its system (OOD: A-primary only)
SETS = {key: {"validation": (32, 6), "long_validation": (4, 60), "long_test": (48, 60)} for key in (P_KEY, R_KEY, B_KEY)}
SETS[P_KEY]["ood"] = (24, 30)
TRAIN_POOL = (256, 6)
H6 = {"scenarios": 8, "designs": 6 * 6 * 5, "sobol": 20,                   # PREREG H6
      "latent_arms": ("U", "U+F", "S"), "seeds": {"U": 5, "U+F": 5, "S": 5, "C": 1, "C+L": 3}}
SESSION_WORK_H = 11.0                                                      # a job checkpoints after 11 h (--hours)
STARTUP_S = 300.0      # per session: clone/kit, install, load; first log lines of downloaded kernels at 141-317 s


def _sens_runs() -> dict:
    out = {}
    for arms, _ in SENS_CELLS.values():
        for a in arms:
            out[a] = out.get(a, 0) + 3
    return out


def n_cells() -> int:
    return sum(n for cols in CELL_DESIGN.values() for _, _, n in cols) + sum(3 * len(a) for a, _ in SENS_CELLS.values())


def ae_specs() -> list:
    """Every autoencoder job: one per (system, material, size, seed) of the learned cells (the most seeds of
    any learned arm in that column), the full-data MLP gate wave (5 d per full-data seed), and one per
    sensitivity kind and seed (degraded data and the next d with the R1 loss, R1's own with plain MSE)."""
    cols = {}
    for arm in WITH_AE:
        for key, size, n in CELL_DESIGN[arm]:
            cols[(key, size)] = max(cols.get((key, size), 0), n)
    sy = lambda key: key.split("_")[0]  # noqa: E731
    out = [{"kind": "study", "system": sy(k), "key": k, "size": s, "seed": i, "weight": 1}
           for (k, s), n in cols.items() for i in range(n)]
    out += [{"kind": "gate", "system": sy(k), "key": k, "size": "full", "seed": i, "weight": 1, "d_index": j}
            for (k, s), n in cols.items() if s == "full" for i in range(n) for j in range(D_GRID_N)]
    for kind, (_, (k, s)) in SENS_CELLS.items():
        out += [{"kind": "sensitivity", "system": sy(k), "key": k, "size": s, "seed": i,
                 "weight": 0 if kind == "aemse" else 1} for i in range(3)]
    return out


def _pool(size) -> int:
    return POOL if size == "full" else int(size)


def ae_hours(t: dict, specs: list, epochs: int = AE_EPOCHS) -> float:
    """60 epochs at the measured full-data epoch time of the system and loss, scaled by the pool size."""
    return sum(epochs * t["ae"][f"{s['system']}|w{s['weight']}"]["epoch_s"] * _pool(s["size"]) / POOL
               for s in specs) / 3600


def cplusl_run_hours(t: dict, iterations: int, K: int) -> float:
    return t["cl_train"][f"K{K}"]["wall_s_per_iter"] * iterations / 3600     # KeyError if K was not timed


def _cl_eval_s_per_traj_cycle(t: dict, K: int) -> float:
    """C+L's rollout, CPU-s per trajectory-cycle, linear in K through the two measured K."""
    e = t["cl_eval"]
    ks = sorted(int(k[1:]) for k in e if k.startswith("K"))
    (k1, k2) = ks[0], ks[-1]
    per = lambda k: e[f"K{k}"]["wall_s"] / (e["n_traj"] * e["cycles"])  # noqa: E731
    return per(k1) + (per(k2) - per(k1)) * (K - k1) / (k2 - k1)


def _step_s(t: dict, h: int) -> float:
    s1, s2 = (t["step_S"][f"h{k}|b{BATCH}"]["wall_s_per_interval"] for k in (1, 2))
    return s1 + (s2 - s1) * (h - 1)


def latent_eval_hours(t: dict, key: str, h: int, n_arms: int) -> float:
    """Batched rollouts (48 trajectories per tensor) of every evaluated set; S's measured batched step is
    used for every latent arm (an upper bound: U's step is cheaper)."""
    s = _step_s(t, h)
    return n_arms * sum(math.ceil(n / BATCH) * cyc * N_SNAP * s for n, cyc in SETS[key].values()) / 3600


def _traj_cycles(key: str) -> int:
    return sum(n * cyc for n, cyc in SETS[key].values())


def solver_eval_hours(t: dict, arm: str, key: str) -> float:
    """C (16 to 128 cells) or MUSCL-C (32, 64): every set's trajectory-cycles at the slowest measured cycle."""
    grids = (16, 32, 64, 128) if arm == "C" else (32, 64)
    return _traj_cycles(key) * sum(max(t[arm][f"{key}|{g}"]["cycle_cpu_s"]) for g in grids) / 3600


def job_sessions(hours: float) -> int:
    return max(1, math.ceil(hours / SESSION_WORK_H))


def session_hours(job_hours: list, startup_s: float = STARTUP_S) -> float:
    """Kaggle session time of jobs run one per session: the work plus a start-up per session."""
    return sum(h + job_sessions(h) * startup_s / 3600 for h in job_hours)


def _ref_s(t: dict, key: str) -> float:
    return max(t["reference"][key]["cycle_cpu_s"])


def need(t: dict, iterations: int, h: int, K: int, n_css: int = 13, startup_s: float = STARTUP_S) -> dict:
    """The confirmatory need in one-run CPU hours per block, and in Kaggle session hours (jobs as
    kaggle/train_jobs.py makes them: one per session, evaluation one job per cell and test set)."""
    blocks, jobs = {}, {}
    # training: latent arms (U-long for S's wall-clock) with their sensitivities, C+L
    runs = {}
    for arm in LATENT:
        for key, size, n in CELL_DESIGN[arm]:
            runs.setdefault(arm, []).extend([(key, size)] * n)
    for kind, (arms, col) in SENS_CELLS.items():
        for a in arms:
            runs[a].extend([col] * 3)
    lat = [_s_per_iteration(t, "S" if a == "U-long" else a, h) * iterations / 3600 for a, cs in runs.items() for _ in cs]
    blocks["training, latent arms"] = {"hours": sum(lat)}
    jobs["training, latent arms"] = lat
    cl = [cplusl_run_hours(t, iterations, K)] * sum(n for _, _, n in CELL_DESIGN["C+L"])
    blocks["training, C+L"] = {"hours": sum(cl)}
    jobs["training, C+L"] = cl
    # autoencoders
    ae = [ae_hours(t, [s]) for s in ae_specs()]
    blocks["autoencoders"] = {"hours": sum(ae)}
    jobs["autoencoders"] = ae
    # evaluation: every latent cell on its validation sets (in its training job) and on each existing test set
    ev_lat, n_eval_jobs = [], 0
    for arm, cs in runs.items():
        for key, size in cs:
            extra = 1 if (arm == "U" and key == P_KEY and size in ("full", 32)) or arm == "U+F" else 0
            ev_lat.append(latent_eval_hours(t, key, h, 1 + extra))
            n_eval_jobs += 3                                   # one job per cell and GATED set (train_jobs)
    blocks["evaluation, latent arms"] = {"hours": sum(ev_lat), "note": "S's batched step for every arm (upper bound)"}
    ev_sol = [solver_eval_hours(t, arm, key) for arm in ("C", "MUSCL-C") for key, _, n in CELL_DESIGN[arm] for _ in range(n)]
    cl_tc = _cl_eval_s_per_traj_cycle(t, K)
    ev_sol += [_traj_cycles(key) * cl_tc / 3600 for key, _, n in CELL_DESIGN["C+L"] for _ in range(n)]
    n_eval_jobs += 3 * sum(n for a in ("C", "MUSCL-C", "C+L") for _, _, n in CELL_DESIGN[a])
    blocks["evaluation, solver arms"] = {"hours": sum(ev_sol)}
    # H6: every design of every scenario rolled out to the cyclic steady state, optima and Sobol by the reference
    sc, de = H6["scenarios"], H6["designs"]
    h6_lat = sum(H6["seeds"][a] for a in H6["latent_arms"]) * math.ceil(de / BATCH) * n_css * N_SNAP * _step_s(t, h)
    h6_c = de * n_css * max(t["C"][f"{P_KEY}|32"]["cycle_cpu_s"])
    h6_cl = H6["seeds"]["C+L"] * de * n_css * cl_tc
    h6_ref = (H6["sobol"] + sum(H6["seeds"].values())) * n_css * _ref_s(t, P_KEY)
    blocks["H6 searches and reference checks"] = {"hours": sc * (h6_lat + h6_c + h6_cl + h6_ref) / 3600,
                                                  "note": f"{n_css} cycles to the cyclic steady state (assumed)"}
    # the confirmatory reference sets (generated after the freeze; the pilot sets are not counted)
    ref = sum((TRAIN_POOL[0] * TRAIN_POOL[1] + _traj_cycles(key)) * _ref_s(t, key) for key in (P_KEY, R_KEY, B_KEY))
    blocks["reference data (confirmatory sets)"] = {"hours": ref / 3600}
    total = sum(b["hours"] for b in blocks.values())
    # session time: one job per session as implemented; evaluation, H6 and data spread over their kernels
    per_job = jobs["training, latent arms"] + jobs["training, C+L"] + jobs["autoencoders"]
    rest = (blocks["evaluation, latent arms"]["hours"] + blocks["evaluation, solver arms"]["hours"]
            + blocks["H6 searches and reference checks"]["hours"] + blocks["reference data (confirmatory sets)"]["hours"])
    n_rest = n_eval_jobs
    kernels = len(per_job) + n_rest
    sessions = sum(job_sessions(x) for x in per_job) + max(n_rest, math.ceil(rest / SESSION_WORK_H))
    sess_h = total + sessions * startup_s / 3600
    return {"iterations": iterations, "h": h, "K": K, "n_css": n_css, "blocks": blocks, "total_run_hours": total,
            "kernels": kernels, "sessions": sessions, "session_hours": sess_h}


def load_inputs(results: str) -> dict:
    """The measured costs: results/timing_ceiling.json (kit after R3, d = 64) where present, the earlier
    kernels otherwise (timing_arms.json at d = 16, timing_concurrency_batch.json), and C+L's two files."""
    rd = lambda f: json.load(open(os.path.join(results, f))) if os.path.exists(os.path.join(results, f)) else None  # noqa: E731
    arms, conc, tc = rd("timing_arms.json"), rd("timing_concurrency_batch.json"), rd("timing_ceiling.json")
    t = {"train": dict(arms["train"]), "step_S": conc["step_S"], "cl_train": rd("timing_cplusl_train.json"),
         "cl_eval": rd("timing_cplusl_eval.json"), "concurrent": conc["concurrent"], "sources": {}}
    t["sources"]["train"] = "timing_arms.json (d = 16)"
    if tc:
        t["train"].update(tc["train"])
        t["step_S"] = tc["step_S"]
        for k in ("C", "MUSCL-C", "ae", "reference"):
            t[k] = tc[k]
        t["sources"]["train"] = f"timing_ceiling.json (d = {tc['settings']['q']['d']}) for its arms"
    return t


def conv_factor(results: str) -> float | None:
    """The conv decoder's training time over the MLP's (decoder_diag.json, MOF-303, 60 epochs)."""
    p = os.path.join(results, "decoder_diag", "decoder_diag.json")
    if not os.path.exists(p):
        return None
    rows = [r for r in json.load(open(p))["rows"] if r["key"] == P_KEY and r.get("epochs") == 60]
    m = [r["train_s"] for r in rows if r["decoder"] == "mlp"]
    c = [r["train_s"] for r in rows if r["decoder"] == "conv"]
    return (sum(c) / len(c)) / (sum(m) / len(m)) if m and c else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--iterations", type=int, default=8000)
    ap.add_argument("--n-css", type=int, default=13)
    a = ap.parse_args(argv)
    t = load_inputs(a.results)
    missing = [k for k in ("C", "MUSCL-C", "ae", "reference") if k not in t]
    if missing:
        print(f"not measured yet: {missing} (run the timing kernel, results/timing_ceiling_kernel.py)")
        return None
    print(f"inputs: latent training from {t['sources']['train']}; batched step, C, MUSCL-C, autoencoders and the "
          "reference from timing_ceiling.json; C+L from timing_cplusl_train.json and timing_cplusl_eval.json")
    print(f"{a.iterations} iterations; {a.n_css} cycles to the cyclic steady state in H6 (assumed); "
          f"{STARTUP_S:.0f} s start-up per Kaggle session; one job per session (kaggle/train_jobs.py)")
    out = {}
    for K in (1, 4):
        for h in (1, 2, 4):
            n = need(t, a.iterations, h, K, a.n_css)
            out[(h, K)] = n
            print(f"\nh = {h}, K = {K}:")
            for name, b in n["blocks"].items():
                print(f"  {name:38s} {b['hours']:8.1f} h" + (f"   ({b['note']})" if "note" in b else ""))
            print(f"  {'total, one-run CPU hours':38s} {n['total_run_hours']:8.1f} h")
            print(f"  {'Kaggle session hours (the ceiling unit)':38s} {n['session_hours']:8.1f} h "
                  f"({n['sessions']} sessions, {n['kernels']} kernels)")
    ae_g = ae_hours(t, [s for s in ae_specs() if s["kind"] == "gate"])
    f = conv_factor(a.results)
    if f:
        print(f"\ncontingency, not in the totals: a conv gate wave (65 autoencoders) at {f:.1f} x the MLP's time "
              f"adds about {ae_g * f:.0f} h")
    b = training_budget(t, a.iterations, 2, cplusl_run_hours(t, a.iterations, 4))
    print("\nreduction order at h = 2, K = 4, training hours saved:")
    for r in b["reductions"]:
        print(f"  {r['step']}: {r['hours']:.1f}")
    return out


if __name__ == "__main__":
    main()
