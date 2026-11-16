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


def _step_s(t: dict, h: int, arm: str = "S") -> float:
    """An arm's measured batched rollout step (48 trajectories), linear in h through h = 1 and 2; S's step
    where the arm's own was not measured (an upper bound for the unstructured arms)."""
    own = t.get("step", {}).get("U" if arm == "U-long" else arm)
    ok = own and "error" not in own and all(own[f"h{k}|b{BATCH}"].get("intervals", 1) > 0 for k in (1, 2))
    src = own if ok else t["step_S"]
    s1, s2 = (src[f"h{k}|b{BATCH}"]["wall_s_per_interval"] for k in (1, 2))
    return s1 + (s2 - s1) * (h - 1)


def latent_eval_hours(t: dict, key: str, h: int, n_arms: int, arm: str = "S") -> float:
    """Batched rollouts (48 trajectories per tensor) of every evaluated set at the arm's batched step."""
    s = _step_s(t, h, arm)
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


# ---- the job list (T8 ceiling review, 2026-11-17) ---------------------------------------------------------
REDUCTION_ORDER = ["degraded-data sensitivity", "U-long", "S-Pi", "latent-dimension sensitivity",
                   "autoencoder-loss sensitivity", "MUSCL-C", "sizes 16 and 128 to 2 seeds", "System B S-L"]
_SENS_STEP = {"degraded-data sensitivity": "noise", "latent-dimension sensitivity": "d+",
              "autoencoder-loss sensitivity": "aemse"}
GATED = ("long_test", "ood")                     # the test sets; a cell is evaluated on those its system has
VALIDATION = ("validation", "long_validation")   # scored in the cell's training job


def design_cells() -> list:
    """Every cell of p3b.train.cells.cells() and sensitivity_cells(), with how many projected arms it is also
    evaluated as (evaluate_also: U+proj for U at A-primary full and 32, U+F+proj for every U+F)."""
    out = []
    for arm, cols in CELL_DESIGN.items():
        for key, size, n in cols:
            also = 1 if (arm == "U" and key == P_KEY and size in ("full", 32)) or arm == "U+F" else 0
            out += [{"arm": arm, "key": key, "size": size, "seed": s, "sens": None, "also": also} for s in range(n)]
    for kind, (arms, (key, size)) in SENS_CELLS.items():
        out += [{"arm": a, "key": key, "size": size, "seed": s, "sens": kind, "also": 0} for a in arms for s in range(3)]
    return out


def reduce_cells(cs: list, steps) -> list:
    """The cells left after the reduction steps (p3b.train.cells.apply_reductions)."""
    def gone(c, step):
        if step in _SENS_STEP:
            return c["sens"] == _SENS_STEP[step]
        if step in ("U-long", "S-Pi", "MUSCL-C"):
            return c["arm"] == step and c["sens"] is None
        if step == "sizes 16 and 128 to 2 seeds":
            return c["size"] in (16, 128) and c["seed"] >= 2
        if step == "System B S-L":
            return c["arm"] == "S-L" and c["key"] == B_KEY
        raise KeyError(step)
    return [c for c in cs if not any(gone(c, s) for s in steps)]


def projected_evaluations(cs: list) -> int:
    return sum(c["also"] for c in cs)


def _ae_from_cells(cs: list) -> list:
    sy = lambda key: key.split("_")[0]  # noqa: E731
    study, gate_seeds, sens = [], {}, []
    for c in cs:
        if c["arm"] not in WITH_AE:
            continue
        if c["sens"]:
            s = {"kind": "sensitivity", "system": sy(c["key"]), "key": c["key"], "size": c["size"], "seed": c["seed"],
                 "weight": 0 if c["sens"] == "aemse" else 1, "sens": c["sens"]}
            if not any(x["sens"] == s["sens"] and x["seed"] == s["seed"] for x in sens):
                sens.append(s)
            continue
        s = {"kind": "study", "system": sy(c["key"]), "key": c["key"], "size": c["size"], "seed": c["seed"], "weight": 1}
        if not any((x["key"], x["size"], x["seed"]) == (s["key"], s["size"], s["seed"]) for x in study):
            study.append(s)
        if c["size"] == "full":
            gate_seeds.setdefault(c["key"], set()).add(c["seed"])
    gate = [{"kind": "gate", "system": sy(k), "key": k, "size": "full", "seed": s, "weight": 1, "d_index": j}
            for k, ss in gate_seeds.items() for s in sorted(ss) for j in range(D_GRID_N)]
    return study + gate + sens


def ae_specs() -> list:
    """Every autoencoder job: one per (system, material, size, seed) of the learned cells, the full-data MLP
    gate wave (5 d per full-data seed), and one per sensitivity kind and seed (degraded data and the next d
    with the R1 loss, R1's own with plain MSE)."""
    return _ae_from_cells(design_cells())


def _set_hours(t: dict, key: str, set_name: str, h: int, arm: str) -> float:
    n, cyc = SETS[key][set_name]
    return math.ceil(n / BATCH) * cyc * N_SNAP * _step_s(t, h, arm) / 3600


def _solver_set_hours(t: dict, arm: str, key: str, set_name: str, K: int) -> float:
    n, cyc = SETS[key][set_name]
    if arm == "C+L":
        return n * cyc * _cl_eval_s_per_traj_cycle(t, K) / 3600
    grids = (16, 32, 64, 128) if arm == "C" else (32, 64)
    return n * cyc * sum(max(t[arm][f"{key}|{g}"]["cycle_cpu_s"]) for g in grids) / 3600


def _t4(t: dict, arm: str, h: int) -> float:
    r = t.get("t4_ratio", {})
    return r.get(f"{'S' if arm == 'U-long' else arm}|h{min(h, 2)}", 1.0)


def jobs(t: dict, cs: list, iterations: int, h: int, K: int, n_css: int = 13, four_threads: bool = False) -> list:
    """The confirmatory jobs, each {"kind", "stage", "hours", "blocks": {block: hours}} at one thread (or, with
    four_threads, the latent training at the measured four-thread cost): reference data (stage 0), the
    autoencoders (1), the training jobs with their validation sets (2), and the evaluation of each existing
    test set per cell, the decoder-only floor and H6 (3)."""
    js = []

    def add(kind, stage, blocks):
        js.append({"kind": kind, "stage": stage, "hours": sum(blocks.values()), "blocks": blocks})
    for key in (P_KEY, R_KEY, B_KEY):                 # reference data: chunks of at most 11 hours
        sets = {"train": TRAIN_POOL, **SETS[key]}
        for name, (n, cyc) in sets.items():
            hrs = n * cyc * _ref_s(t, key) / 3600
            k = job_sessions(hrs)
            for _ in range(k):
                add("data", 0, {"reference data (confirmatory sets)": hrs / k})
    for s in _ae_from_cells(cs):
        add("autoencoder", 1, {"autoencoders": ae_hours(t, [s])})
    for c in cs:
        arm, key = c["arm"], c["key"]
        val = VALIDATION
        if arm in LATENT:
            tr = _s_per_iteration(t, "S" if arm == "U-long" else arm, h) * iterations / 3600
            tr *= _t4(t, arm, h) if four_threads else 1.0
            ev = sum(_set_hours(t, key, s, h, arm) + c["also"] * _set_hours(t, key, s, h, "S") for s in val)
            add("training", 2, {"training, latent arms": tr, "evaluation, latent arms": ev})
            for s in GATED:
                if s in SETS[key]:
                    add("evaluation", 3, {"evaluation, latent arms": _set_hours(t, key, s, h, arm)
                                          + c["also"] * _set_hours(t, key, s, h, "S")})
        else:                                          # C, MUSCL-C, C+L
            blocks = {"evaluation, solver arms": sum(_solver_set_hours(t, arm, key, s, K) for s in val)}
            if arm == "C+L":
                blocks["training, C+L"] = cplusl_run_hours(t, iterations, K)
            add("training", 2, blocks)
            for s in GATED:
                if s in SETS[key]:
                    add("evaluation", 3, {"evaluation, solver arms": _solver_set_hours(t, arm, key, s, K)})
    # the decoder-only floor (R1; registered, never removed): until its own cost is measured, one projected
    # batched rollout of the long test and OOD sets per A-primary study autoencoder, at S's step
    for s in _ae_from_cells(cs):
        if s["kind"] == "study" and s["key"] == P_KEY:
            add("floor", 3, {"decoder-only floor": sum(_set_hours(t, P_KEY, x, h, "S") for x in GATED)})
    # H6: per scenario, each latent model's search (180 designs batched), C's and each C+L seed's, and the
    # reference checks of the 20 Sobol points and the arms' optima, to the cyclic steady state
    de = H6["designs"]
    for _ in range(H6["scenarios"]):
        for a in H6["latent_arms"]:
            for _s in range(H6["seeds"][a]):
                add("H6", 3, {"H6 searches and reference checks":
                              math.ceil(de / BATCH) * n_css * N_SNAP * _step_s(t, h, a) / 3600})
        add("H6", 3, {"H6 searches and reference checks": de * n_css * max(t["C"][f"{P_KEY}|32"]["cycle_cpu_s"]) / 3600})
        for _s in range(H6["seeds"]["C+L"]):
            add("H6", 3, {"H6 searches and reference checks": de * n_css * _cl_eval_s_per_traj_cycle(t, K) / 3600})
        add("H6", 3, {"H6 searches and reference checks":
                      (H6["sobol"] + sum(H6["seeds"].values())) * n_css * _ref_s(t, P_KEY) / 3600})
    return js


def packed_sessions(js: list, slowdown: float, startup_s: float = STARTUP_S, size: int = 4) -> dict:
    """Session time with the jobs packed `size` to a kernel per stage, in order (kaggle/run_queue.py
    pack_jobs): a kernel lasts as long as its slowest member, each member slowed by `slowdown` when four
    run at once; past 11 hours of work it continues in a further session; each session pays a start-up."""
    kernels = sessions = 0
    hours = 0.0
    for stage in sorted({j["stage"] for j in js}):
        g = [j["hours"] for j in js if j["stage"] == stage]
        for a in range(0, len(g), size):
            wall = max(g[a:a + size]) * slowdown
            n = job_sessions(wall)
            kernels += 1
            sessions += n
            hours += wall + n * startup_s / 3600
    return {"kernels": kernels, "sessions": sessions, "session_hours": hours}


def single_sessions(js: list, startup_s: float = STARTUP_S) -> dict:
    return {"kernels": len(js), "sessions": sum(job_sessions(j["hours"]) for j in js),
            "session_hours": session_hours([j["hours"] for j in js], startup_s)}


def need(t: dict, iterations: int, h: int, K: int, n_css: int = 13, startup_s: float = STARTUP_S,
         reductions=()) -> dict:
    """The confirmatory need (or, with `reductions`, the need left after those steps) in one-run CPU hours
    per block and in Kaggle session hours, the ceiling's unit, for the implemented layout (one-thread jobs
    packed four per session) and for one job per session at the measured four-thread cost."""
    cs = reduce_cells(design_cells(), reductions)
    js = jobs(t, cs, iterations, h, K, n_css)
    blocks = {}
    for j in js:
        for b, v in j["blocks"].items():
            blocks.setdefault(b, {"hours": 0.0})["hours"] += v
    js4 = jobs(t, cs, iterations, h, K, n_css, four_threads=True)
    single = single_sessions(js4, startup_s)
    single["training_hours"] = sum(j["blocks"].get("training, latent arms", 0.0) for j in js4)
    return {"iterations": iterations, "h": h, "K": K, "n_css": n_css, "reductions": list(reductions),
            "blocks": blocks, "total_run_hours": sum(b["hours"] for b in blocks.values()), "jobs": len(js),
            "packed": packed_sessions(js, t.get("pack_slowdown", 1.0), startup_s), "single": single}


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
        for k in ("C", "MUSCL-C", "ae", "reference", "step"):
            t[k] = tc[k]
        t["sources"]["train"] = f"timing_ceiling.json (d = {tc['settings']['q']['d']}) for its arms"
        # four threads over one (timing_arms.json, d = 16), for pricing one job per session at four threads
        t["t4_ratio"] = {f"{a}|h{k}": arms["train"][f"{a}|h{k}|t4"]["wall_s_per_iteration"]
                         / arms["train"][f"{a}|h{k}|t1"]["wall_s_per_iteration"]
                         for a in {x.split("|")[0] for x in arms["train"]} for k in (1, 2)
                         if f"{a}|h{k}|t4" in arms["train"]}
        # four one-thread jobs at once: each job's slowdown, the larger of U's and S's (concurrency timing)
        c = conc["concurrent"]
        if c:
            t["pack_slowdown"] = max(sum(c[f"{a}|h2|n4"]["wall_s_per_iteration"]) / 4 / c[f"{a}|h2|n1"]["wall_s_per_iteration"][0]
                                     for a in ("U", "S"))
        ref = rd("timing_reference.json")
        if ref:                                  # the probe timed one point per key: the earlier records stay a floor
            a_max = ref["data_cost_decision"]["system_A_2048_exact_max"]
            nz_b = ref["data_cost_decision"]["N_z_B"]
            b_max = max([r["cpu_s"] for r in ref["records"] if r["system"] == "B" and r["N_z"] == nz_b and r.get("ok")],
                        default=0.0)
            t["reference"] = {k: {**v, "cycle_cpu_s": list(v["cycle_cpu_s"]) + [b_max if k == B_KEY else a_max]}
                              for k, v in t["reference"].items()}
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
    print(f"inputs: latent training from {t['sources']['train']}; batched steps, C, MUSCL-C, autoencoders and the "
          "reference from timing_ceiling.json; C+L from timing_cplusl_train.json and timing_cplusl_eval.json")
    print(f"{a.iterations} iterations; {a.n_css} cycles to the cyclic steady state in H6 (assumed); "
          f"{STARTUP_S:.0f} s start-up per Kaggle session (a typical estimate: first log lines at 141 to 317 s)")
    print(f"implemented layout: one-thread jobs packed four per session (kaggle/run_queue.py pack_jobs), each "
          f"member {t.get('pack_slowdown', 1.0):.2f} x slower than alone (timing_concurrency_batch.json, d = 16 "
          "training); alternative: one job per session, latent training at the four-thread cost (timing_arms.json)")
    out = {}
    for K in (1, 4):
        for h in (1, 2, 4):
            n = need(t, a.iterations, h, K, a.n_css)
            c = need(t, a.iterations, h, K, a.n_css, reductions=REDUCTION_ORDER)
            out[(h, K)] = (n, c)
            print(f"\nh = {h}, K = {K} ({n['jobs']} jobs):")
            for name, b in n["blocks"].items():
                print(f"  {name:38s} {b['hours']:8.1f} h")
            print(f"  {'total, one-run CPU hours':38s} {n['total_run_hours']:8.1f} h")
            for lay, label in (("packed", "packed four per session"), ("single", "one job per session")):
                print(f"  session hours, {label:24s} {n[lay]['session_hours']:8.1f} h ({n[lay]['sessions']} sessions, "
                      f"{n[lay]['kernels']} kernels); core after every reduction {c[lay]['session_hours']:8.1f} h")
    ae_g = ae_hours(t, [s for s in ae_specs() if s["kind"] == "gate"])
    f = conv_factor(a.results)
    if f:
        print(f"\ncontingency, not in the totals: a conv gate wave (65 autoencoders) at {f:.1f} x the MLP's time "
              f"adds about {ae_g * f:.0f} h of work")
    return out


if __name__ == "__main__":
    main()
