"""The study's cells, their Kaggle job names, the compute ceiling, and one cell's training and
evaluation (PREREG Section 0 design table, Sections 3, 4, 6 and 11; docs/compute_3B.md Section 3;
Plan 4, Task 4, revised after the plan 4 review).

A cell is (arm, system, material, training size, seed). U+proj and U+F+proj have no cells of their
own: they reuse U's and U+F's trained weights and are evaluated with them (`evaluate_also`). All
learned latent arms of one (system, material, size, seed) share one frozen autoencoder, trained by
its own job. Training saves the weights and evaluates only the validation sets; test sets open per
system and material after H3a passed (evaluate_cell).
"""
from __future__ import annotations

import glob
import json
import os
import shutil
from dataclasses import dataclass

import torch

PRIMARY = ("A", "MOF-303")
REPL = ("A", "Al-fumarate")
SYS_B = ("B", None)

# (arm, [(system, material, size, seeds), ...]), from the PREREG Section 0 design table
DESIGN = [
    ("U", [(*PRIMARY, "full", 5), (*PRIMARY, 32, 5), (*PRIMARY, 16, 3), (*PRIMARY, 64, 3), (*PRIMARY, 128, 3),
           (*REPL, "full", 3), (*REPL, 32, 3), (*SYS_B, "full", 5), (*SYS_B, 32, 5)]),
    ("U+F", [(*PRIMARY, "full", 5), (*PRIMARY, 32, 5), (*REPL, "full", 3), (*REPL, 32, 3),
             (*SYS_B, "full", 5), (*SYS_B, 32, 5)]),
    ("U-long", [(*PRIMARY, "full", 3)]),
    ("P", [(*PRIMARY, "full", 5), (*PRIMARY, 32, 5), (*PRIMARY, 16, 3), (*PRIMARY, 64, 3), (*PRIMARY, 128, 3),
           (*REPL, "full", 3), (*REPL, 32, 3)]),
    ("S", [(*PRIMARY, "full", 5), (*PRIMARY, 32, 5), (*PRIMARY, 16, 3), (*PRIMARY, 64, 3), (*PRIMARY, 128, 3),
           (*REPL, "full", 3), (*REPL, 32, 3), (*SYS_B, "full", 5), (*SYS_B, 32, 5)]),
    ("S-L", [(*PRIMARY, "full", 5), (*PRIMARY, 32, 5), (*SYS_B, "full", 3), (*SYS_B, 32, 3)]),
    ("S-M", [(*PRIMARY, "full", 5)]),
    ("S-Pi", [(*PRIMARY, "full", 5)]),
    ("FNO", [(*PRIMARY, "full", 3), (*PRIMARY, 32, 3)]),
    ("C", [(*PRIMARY, "full", 1), (*REPL, "full", 1), (*SYS_B, "full", 1)]),
    ("C+L", [(*PRIMARY, "full", 3), (*PRIMARY, 32, 3)]),
    ("MUSCL-C", [(*PRIMARY, "full", 1)]),
]
U_PROJ_COLUMNS = {(*PRIMARY, "full"), (*PRIMARY, 32)}

# docs/compute_3B.md Section 3, in order; the descriptive sensitivities are separate jobs (no cells)
REDUCTION_ORDER = ["degraded-data sensitivity", "U-long", "S-Pi", "latent-dimension sensitivity", "MUSCL-C",
                   "sizes 16 and 128 to 2 seeds", "System B S-L"]


@dataclass(frozen=True)
class Cell:
    arm: str
    system: str
    material: str | None
    size: object
    seed: int
    evaluate_also: tuple = ()


def cells() -> list:
    out = []
    for arm, cols in DESIGN:
        for system, material, size, n in cols:
            also = ()
            if arm == "U" and (system, material, size) in U_PROJ_COLUMNS:
                also = ("U+proj",)
            elif arm == "U+F":
                also = ("U+F+proj",)
            out += [Cell(arm, system, material, size, s, also) for s in range(n)]
    return out


_ARM = {"U": "u", "U+F": "uf", "U-long": "ulong", "P": "p", "S": "s", "S-L": "sl", "S-M": "sm", "S-Pi": "spi",
        "FNO": "fno", "C": "c", "C+L": "cl", "MUSCL-C": "muscl"}
_MAT = {"MOF-303": "-m303", "Al-fumarate": "-alf", None: ""}
_SENS = {"noise": "noise", "d+": "dplus"}


def job_name(c: Cell) -> str:
    if "|" in c.arm:                                  # a sensitivity cell, e.g. "U|noise"
        base, kind = c.arm.split("|")
        return f"p3b-{_ARM[base]}{_SENS[kind]}-{c.system.lower()}{_MAT[c.material]}-{c.size}-s{c.seed}"
    return f"p3b-{_ARM[c.arm]}-{c.system.lower()}{_MAT[c.material]}-{c.size}-s{c.seed}"


def ae_job_name(c: Cell) -> str:
    """The job that trains the autoencoder this cell shares (per system, material, size, seed)."""
    return f"p3b-ae-{c.system.lower()}{_MAT[c.material]}-{c.size}-s{c.seed}"


SOLVER_ARMS = ("C", "MUSCL-C", "C+L")


def _removed_by(step: str, c: Cell) -> bool:
    if step == "U-long":
        return c.arm == "U-long"
    if step == "S-Pi":
        return c.arm == "S-Pi"
    if step == "MUSCL-C":
        return c.arm == "MUSCL-C"
    if step == "sizes 16 and 128 to 2 seeds":
        return c.size in (16, 128) and c.seed >= 2
    if step == "System B S-L":
        return c.arm == "S-L" and c.system == "B"
    return False                                     # the sensitivities have no cells here




# ---- sensitivities and the ceiling's fixed costs ----------------------------------------------------
def sensitivity_cells() -> list:
    """The descriptive sensitivities (PREREG Section 6) as cells for the ceiling model: degraded
    data (1 % multiplicative noise on the stored fields, size 32; U, U+F, S; 3 seeds) and the next
    passing latent dimension (U, U+F, S; full data; 3 seeds)."""
    out = []
    for arm in ("U", "U+F", "S"):
        out += [Cell(f"{arm}|noise", *PRIMARY, 32, s) for s in range(3)]
        out += [Cell(f"{arm}|d+", *PRIMARY, "full", s) for s in range(3)]
    return out


def _removed_by_sens(step: str, c: Cell) -> bool:
    return (step == "degraded-data sensitivity" and c.arm.endswith("|noise")) or \
        (step == "latent-dimension sensitivity" and c.arm.endswith("|d+"))


def apply_reductions(cs: list, steps: list) -> list:
    return [c for c in cs if not any(_removed_by(s, c) or _removed_by_sens(s, c) for s in steps)]


def plan_reductions(spent_h: float, remaining: list, cost_h: dict, ceiling_h: float = 700.0,
                    fixed_h: float = 0.0) -> list:
    """The reduction steps, in order, needed so that spent + fixed costs (autoencoder and H6 jobs)
    + the projected cost of the remaining cells fits under the ceiling; all seven if even that does
    not fit (the never-reduced core then runs and the excess is reported)."""
    steps = []
    for step in [None] + REDUCTION_ORDER:
        if step is not None:
            steps.append(step)
        if spent_h + fixed_h + sum(cost_h[c] for c in apply_reductions(remaining, steps)) <= ceiling_h:
            return steps
    return steps


# ---- one cell ------------------------------------------------------------------------------------------
DESIGN_NAME = {("A", "MOF-303"): "A_primary", ("A", "Al-fumarate"): "A_replication", ("B", None): "B"}
LEARNED_WITH_AE = ("U", "U+F", "U-long", "P", "S", "S-L", "S-M", "S-Pi")
VALIDATION_SETS = ("validation", "long_validation")
GATED_SETS = ("test", "long_test", "ood")


def _physics(system, material, ds):
    from p3b.latent.physics import ColumnPhysics, StorePhysics
    if system == "A":
        from p3b.simulate_a import setup_a
        return ColumnPhysics(setup_a(ds.meta[0]["point"], material, ds.n_cells)[0])
    from p3b.simulate_b import T_AMB, make_store
    return StorePhysics(make_store(ds.n_cells), T_AMB)


def _sha(path) -> str:
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _set_files(data_dir, name):
    from p3b.gates import load_meta
    files = glob.glob(os.path.join(data_dir, name, "*.npz"))
    return sorted(files, key=lambda p: int(load_meta(p)["point"]["index"]))


def _train_files(data_dir: str, size, pool_size: int | None = None) -> tuple:
    """The training files: the whole pool for "full", else its first `size` trajectories by design
    index (nested subsets, p3b.design.train_subset); both checked against the pool."""
    from p3b.gates import load_meta
    files = _set_files(data_dir, "train")
    idx = [int(load_meta(p)["point"]["index"]) for p in files]
    if pool_size is not None and len(files) != pool_size:
        raise ValueError(f"the train pool has {len(files)} trajectories, the design {pool_size}")
    if idx != list(range(len(idx))):
        raise ValueError(f"the train pool is not the design's first trajectories: {idx[:8]}...")
    if size != "full":
        files, idx = files[:int(size)], idx[:int(size)]
    return files, idx


def _pool_size(cell, q):
    if "pool_size" in q:
        return q["pool_size"]
    from p3b.design import SET_SIZES
    return SET_SIZES[(DESIGN_NAME[(cell.system, cell.material)], "train")][0]


def matched_widths(d: int, k: int, system: str, n_cells: int, n_ch: int, n_out: int, width_s: int = 128) -> dict:
    """Widths of U, U+F, P and FNO whose trainable dynamics parameters match S's within +-5 %
    (PREREG Section 4); S, S-L, S-M, S-Pi keep the reference width."""
    from p3b.latent.arms_structured import SDynamics
    from p3b.latent.arms_unstructured import UDynamics, UFDynamics, count_params, match_width
    from p3b.latent.fno import FNO1d
    with torch.device("meta"):
        target = count_params(SDynamics(d, k, None, kappa=1.0, width=width_s))
    w_u = match_width(lambda w: UDynamics(d, k, w), target)
    return {"S": width_s, "S-L": width_s, "S-M": width_s, "S-Pi": width_s, "U": w_u, "U-long": w_u, "P": w_u,
            "U+F": match_width(lambda w: UFDynamics(d, k, w, system), target),
            "FNO": match_width(lambda w: FNO1d(n_cells, n_ch, n_out, k, width=w), target), "_S_params": target}


# ---- the full-data decoder gate (PREREG 4, 5.1; plan 2026-11-11-p3b-full-data-decoder-gate) -------------
D_GRID = (16, 24, 32, 48, 64)


def gate_specs(cs, kind: str, grid=D_GRID) -> list:
    """One gate autoencoder per (system, material, d, seed of the full-data autoencoders) for one decoder kind."""
    seeds = {}
    for c in cs:
        if c.size == "full" and c.arm in LEARNED_WITH_AE:
            seeds.setdefault((c.system, c.material), set()).add(c.seed)
    return [{"name": f"p3b-gate-{system.lower()}{_MAT[material]}-{kind}-d{d}-s{s}", "system": system,
             "material": material, "kind": kind, "d": int(d), "seed": int(s)}
            for (system, material), ss in seeds.items() for d in grid for s in sorted(ss)]


def _cycles_from(ds, first: int):
    """The Dataset restricted to cycles first, first + 1, ... (long validation at cycles 40 to 60: first = 39)."""
    import dataclasses
    kw = {f.name: getattr(ds, f.name) for f in dataclasses.fields(ds)}
    for name in ("X", "raw_inputs", "features", "t", "N_true", "E_true", "W_ref", "wall_abs"):
        if kw[name] is not None:
            kw[name] = kw[name][:, first:]
    return type(ds)(**kw)


def gate_ae(spec: dict, data_dir: str, out_dir: str, css_cycle: int, q: dict | None = None) -> str:
    """Train one full-data autoencoder of the spec and score the decoder gate on validation, and on long
    validation at cycles 40 to 60 when the pilot's cyclic steady state is after cycle 6 (PREREG 5.1).
    Writes gate_<name>.json and the weights; returns the json path."""
    from p3b.latent.autoencoder import decoder_gate, make_autoencoder, train_autoencoder
    from p3b.latent.data import Standardiser, load_set
    from p3b.pilot.rules import pilot_key
    q = q or {}
    c = Cell("gate", spec["system"], spec["material"], "full", spec["seed"])
    files, _ = _train_files(data_dir, "full", _pool_size(c, q))
    ds = load_set(files)
    n_ch, n_out = (3, 2) if c.system == "A" else (2, 1)
    std = Standardiser.fit(ds.X, ds.n_cells, n_ch, n_out, features=ds.features)
    torch.manual_seed(c.seed)
    ae = make_autoencoder(spec["kind"], ds.n_cells, n_ch, n_out, int(spec["d"]), width=q.get("ae_width", 256))
    train_autoencoder(ae, std.encode(ds.X), epochs=q.get("ae_epochs", 60), seed=c.seed)
    os.makedirs(out_dir, exist_ok=True)
    w = os.path.join(out_dir, f"{spec['name']}.pt")
    torch.save(ae.state_dict(), w)
    del ds
    val = load_set(_set_files(data_dir, "validation"))
    g = decoder_gate(ae, val, std, _physics(c.system, c.material, val))
    res = {**spec, "key": pilot_key(c.system, c.material), "n_train": len(files), "gate": g, "gate_long": None,
           "passed": bool(g["passed"]), "css_cycle": int(css_cycle), "weights_sha256": _sha(w)}
    if css_cycle > 6:
        lv = _cycles_from(load_set(_set_files(data_dir, "long_validation")), q.get("long_from", 39))
        res["gate_long"] = decoder_gate(ae, lv, std, _physics(c.system, c.material, lv))
        res["passed_long"] = bool(res["gate_long"]["passed"])
    path = os.path.join(out_dir, f"gate_{spec['name']}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    return path


def _autoencoder_for(pilot: dict, n_state: int, d: int, n_cells: int, system: str, q: dict):
    """The cell's autoencoder of the kind the pilot recorded for its system and material (PREREG 4)."""
    from p3b.latent.autoencoder import make_autoencoder
    n_ch, n_out = (3, 2) if system == "A" else (2, 1)
    assert n_state == n_cells * n_ch + n_out, (n_state, n_cells, n_ch, n_out)
    return make_autoencoder(pilot.get("decoder", "mlp"), n_cells, n_ch, n_out, d, width=q.get("ae_width", 256))


def _ae_file(cell) -> str:
    return f"ae_{cell.system}_{cell.material}_{cell.size}_s{cell.seed}.pt"


def _copy_session(src: str, out_dir: str, names) -> None:
    """Copy the named files of an earlier session's output (searched recursively) into out_dir; a name
    found more than once is an error (which session would be continued is then ambiguous)."""
    for name in names:
        hits = glob.glob(os.path.join(src, "**", name), recursive=True)
        if len(hits) > 1:
            raise RuntimeError(f"{name} found {len(hits)} times under {src}")
        if hits:
            shutil.copy2(hits[0], os.path.join(out_dir, name))


def _load_ae(cell, ae, out_dir, pilot, q, std, ds):
    """The cell's shared autoencoder: from its own autoencoder job (kernel inputs) or out_dir. A
    missing one is an error; only quick test runs may train it here."""
    from p3b.latent.autoencoder import train_autoencoder
    ae_file = _ae_file(cell)
    root = pilot.get("ae_input_root", "/kaggle/input")
    found = [p for p in [os.path.join(out_dir, ae_file)]
             + glob.glob(os.path.join(root, "*", "**", ae_file), recursive=True) if os.path.exists(p)]
    if found:
        ae.load_state_dict(torch.load(found[0]))
        return {"file": ae_file, "source": found[0], "sha256": _sha(found[0])}
    if not q or q.get("require_ae"):
        raise FileNotFoundError(f"shared autoencoder {ae_file} not found (its autoencoder job must run first)")
    train_autoencoder(ae, std.encode(ds.X), epochs=q.get("ae_epochs", 60), seed=cell.seed)
    path = os.path.join(out_dir, ae_file)
    torch.save(ae.state_dict(), path)
    return {"file": ae_file, "source": "trained here (quick)", "sha256": _sha(path)}


def _build(cell, pilot, q, ds, out_dir):
    """(Arm, evaluate-also arms, meta parts) for a learned cell, with the autoencoder loaded."""
    from p3b.latent.arms_structured import SDynamics, SLDynamics
    from p3b.latent.arms_unstructured import UDynamics, UFDynamics
    from p3b.latent.data import Standardiser
    from p3b.latent.fno import FNO1d
    from p3b.latent.functionals import LatentFunctionals
    from p3b.train.trainer import Arm, TermsStandardiser, time_scale
    A = cell.system == "A"
    n_ch, n_out = (3, 2) if A else (2, 1)
    nc, n_state, k = ds.n_cells, ds.X.shape[-1], ds.features.shape[-1]
    std = Standardiser.fit(ds.X, nc, n_ch, n_out, features=ds.features)
    d = int(pilot["d"])
    widths = ({a: q["width"] for a in ("U", "U-long", "P", "U+F", "FNO", "S", "S-L", "S-M", "S-Pi")}
              if "width" in q else matched_widths(d, k, cell.system, nc, n_ch, n_out))
    width = widths[cell.arm]
    kind = "U" if cell.arm == "U-long" else cell.arm
    phys = _physics(cell.system, cell.material, ds)
    ae_meta, ae, lf, terms = None, None, None, None
    torch.manual_seed(cell.seed)
    if kind != "FNO":
        ae = _autoencoder_for(pilot, n_state, d, nc, cell.system, q)
        ae_meta = _load_ae(cell, ae, out_dir, pilot, q, std, ds)
        lf = LatentFunctionals(ae.decode, std, phys, cell.system, nc)
    ts = time_scale(ds)
    torch.manual_seed(cell.seed)
    if kind in ("U", "P"):
        dyn = UDynamics(d, k, width, time_scale=ts)
    elif kind == "U+F":
        dyn = UFDynamics(d, k, width, cell.system, time_scale=ts)
        terms = TermsStandardiser.fit(lf, ae, std, ds)
    elif kind in ("S", "S-M", "S-Pi"):
        dyn = SDynamics(d, k, lf, kappa=pilot["kappa"], width=width, metriplectic=kind != "S-M",
                        port_projection=kind != "S-Pi", time_scale=ts)
    elif kind == "S-L":
        dyn = SLDynamics(d, k, lf, kappa=pilot["kappa"], width=width, time_scale=ts)
    elif kind == "FNO":
        dyn = FNO1d(nc, n_ch, n_out, k, width=width)
    else:
        raise ValueError(f"unknown arm {cell.arm!r}")
    arm = Arm(kind, dyn, ae, std, lf, terms=terms)
    meta = {"autoencoder": ae_meta, "time_scale": ts, "width": width,
            "widths_matched_to_S_params": widths.get("_S_params")}
    return arm, phys, terms, meta


def _weights_file(cell):
    return job_name(cell) + "_weights.pt"


def train_cell(cell: Cell, data_dir: str, out_dir: str, pilot: dict, quick: dict | None = None,
               deadline: float | None = None, resume_from: str | None = None):
    """Train the cell, save its weights (and sha256), and evaluate it on the validation sets only
    (validation, long validation: H3a is checked there before any test set is opened). Writes
    out_dir/<job name>.json and returns its path.

    Sessions: past `deadline` (time.monotonic()) training checkpoints and stops, and out_dir/RESUME.json
    is written and returned instead. `resume_from`: an earlier session's output (searched recursively,
    as Kaggle mounts it), whose checkpoint and autoencoder are copied in and continued."""
    from pathlib import Path

    from p3b.eval.evaluate import write_results
    from p3b.latent.data import load_set
    from p3b.train.trainer import TrainConfig, train
    q = quick or {}
    os.makedirs(out_dir, exist_ok=True)
    if cell.arm in SOLVER_ARMS:
        return _solver_cell(cell, data_dir, out_dir, pilot, q, VALIDATION_SETS, train=True, deadline=deadline,
                            resume_from=resume_from)
    ckpt = os.path.join(out_dir, job_name(cell) + "_ckpt.pt")
    if resume_from:
        _copy_session(resume_from, out_dir, [os.path.basename(ckpt), _ae_file(cell)])
    train_files, train_idx = _train_files(data_dir, cell.size, _pool_size(cell, q))
    ds = load_set(train_files)
    arm, phys, terms, meta = _build(cell, pilot, q, ds, out_dir)
    if cell.arm == "U-long":
        if "iterations_ulong" not in pilot:
            raise KeyError("iterations_ulong (S's measured wall-clock) is required for U-long")
        iters = pilot["iterations_ulong"]
    else:
        iters = pilot["iterations"]
    hist = train(arm, ds, TrainConfig(lr=pilot["lr"][cell.arm], iterations=iters, seed=cell.seed,
                                      h_steps=pilot["h_steps"], n_windows=q.get("n_windows", 8),
                                      n_steps=q.get("n_steps", 20), checkpoint=ckpt, deadline=deadline))
    if "stopped_at" in hist:                                   # the session ends; a later one resumes
        path = Path(out_dir) / "RESUME.json"
        path.write_text(json.dumps({"job": job_name(cell), "stopped_at": hist["stopped_at"], "iterations": iters,
                                    "checkpoint": os.path.basename(ckpt)}, indent=1))
        return path
    wpath = os.path.join(out_dir, _weights_file(cell))
    torch.save(arm.dyn.state_dict(), wpath)
    rows = _evaluate_sets(cell, arm, phys, terms, data_dir, pilot, q, VALIDATION_SETS)
    meta.update({"arm": cell.arm, "system": cell.system, "material": cell.material, "size": cell.size,
                 "seed": cell.seed, "params": hist["params"], "pilot": pilot, "iterations": iters,
                 "resumed_from": hist.get("resumed_from"),
                 "train_loss_last": hist["loss"][-1] if hist["loss"] else None,
                 "train_skipped": hist["skipped"], "train_skipped_grad": hist["skipped_grad"],
                 "train_cpu_s": hist["wall_s"], "flags": hist["flags"], "train_indices": train_idx,
                 "weights": {"file": _weights_file(cell), "sha256": _sha(wpath)}})
    path = os.path.join(out_dir, job_name(cell) + ".json")
    write_results(path, rows, meta)
    return Path(path)


def _evaluate_sets(cell, arm, phys, terms, data_dir, pilot, q, sets):
    from p3b.eval.evaluate import evaluate
    from p3b.gates import load_meta
    from p3b.latent.data import load_set
    from p3b.train.trainer import Arm
    rows = []
    for set_name in sets:
        files = _set_files(data_dir, set_name)
        if not files:
            continue
        ds = load_set(files)
        n_cycles = q.get("n_cycles", ds.X.shape[1])
        for name in (cell.arm,) + tuple(cell.evaluate_also):
            ev_kind = {"U+proj": "U+proj", "U+F+proj": "U+F+proj"}.get(name, arm.kind)
            ev = Arm(ev_kind, arm.dyn, arm.ae, arm.std, arm.lf, terms=terms)
            for r in evaluate(ev, ds, phys, n_cycles, pilot["h_steps"], pilot["floors"], pilot["tau_S"], cell.seed):
                r.update(arm=name, set=set_name, design_index=int(load_meta(files[r["trajectory"]])["point"]["index"]))
                rows.append(r)
    return rows


def evaluate_cell(cell: Cell, set_name: str, data_dir: str, out_dir: str, pilot: dict, gate: dict,
                  quick: dict | None = None):
    """Evaluate a trained cell (its saved weights) on one named set. Test sets (test, long test,
    OOD) open only after H3a passed for the cell's system and material (`gate`, from the H3a
    check on the validation sets); otherwise PermissionError. Writes out_dir/<job name>_<set>.json."""
    from pathlib import Path

    from p3b.eval.evaluate import write_results
    from p3b.latent.data import load_set
    from p3b.pilot.rules import pilot_key
    q = quick or {}
    key = pilot_key(cell.system, cell.material)
    if set_name in GATED_SETS and gate.get(key) != "pass":
        raise PermissionError(f"H3a has not passed for {key}: the {set_name} set stays closed")
    if cell.arm in SOLVER_ARMS:
        return _solver_cell(cell, data_dir, out_dir, pilot, q, (set_name,), train=False)
    train_files, _ = _train_files(data_dir, cell.size, _pool_size(cell, q))
    ds = load_set(train_files)
    arm, phys, terms, meta = _build(cell, pilot, q, ds, out_dir)
    wpath = os.path.join(out_dir, _weights_file(cell))
    arm.dyn.load_state_dict(torch.load(wpath))
    rows = _evaluate_sets(cell, arm, phys, terms, data_dir, pilot, q, (set_name,))
    meta.update({"arm": cell.arm, "system": cell.system, "material": cell.material, "size": cell.size,
                 "seed": cell.seed, "set": set_name, "pilot": pilot,
                 "weights": {"file": _weights_file(cell), "sha256": _sha(wpath)}})
    path = os.path.join(out_dir, f"{job_name(cell)}_{set_name}.json")
    write_results(path, rows, meta)
    return Path(path)


def _solver_cell(cell, data_dir, out_dir, pilot, q, sets, train: bool, deadline: float | None = None,
                 resume_from: str | None = None):
    """C (16-128 cells) and MUSCL-C (32, 64): deterministic runs on each set's trajectories. C+L:
    trained once (multipliers saved first), then rolled out like C. A failed run is a failure at
    cycle 1 (capped), never a lost cell. C+L's training spans sessions like train_cell's: past
    `deadline` it checkpoints and RESUME.json is returned; `resume_from` continues it."""
    from pathlib import Path

    from p3b.eval.evaluate import evaluate_files, write_results
    from p3b.gates import load_meta
    from p3b.latent.data import load_set
    from p3b.solvers.c_plus_l import Multipliers
    from p3b.solvers.coarse import run_c
    meta_extra = {}
    mult = None
    if cell.arm == "C+L":
        from p3b.train.cplusl import train_cplusl
        mfile = os.path.join(out_dir, _weights_file(cell))
        torch.manual_seed(cell.seed)
        mult = Multipliers(n_out=3)
        if train:
            from pathlib import Path
            ckpt = os.path.join(out_dir, job_name(cell) + "_ckpt.pt")
            if resume_from:
                _copy_session(resume_from, out_dir, [os.path.basename(ckpt)])
            train_files, train_idx = _train_files(data_dir, cell.size, _pool_size(cell, q))
            hist = train_cplusl(mult, load_set(train_files), cell.material, n_coarse=q.get("n_coarse", 32),
                                lr=pilot["lr"]["C+L"], iterations=pilot["iterations"], seed=cell.seed,
                                n_windows=q.get("n_windows", 8), n_steps=q.get("n_steps", 20), K=pilot["K"],
                                checkpoint=ckpt, deadline=deadline)
            if "stopped_at" in hist:                               # the session ends; a later one resumes
                path = Path(out_dir) / "RESUME.json"
                path.write_text(json.dumps({"job": job_name(cell), "stopped_at": hist["stopped_at"],
                                            "iterations": pilot["iterations"],
                                            "checkpoint": os.path.basename(ckpt)}, indent=1))
                return path
            torch.save(mult.state_dict(), mfile)                 # saved before any rollout can fail
            meta_extra = {"train_indices": train_idx, "train_skipped": hist["skipped"],
                          "resumed_from": hist.get("resumed_from"),
                          "train_skipped_grad": hist["skipped_grad"],
                          "params": sum(p.numel() for p in mult.parameters()),
                          "weights": {"file": os.path.basename(mfile), "sha256": _sha(mfile)}}
        else:
            mult.load_state_dict(torch.load(mfile))
            meta_extra = {"weights": {"file": os.path.basename(mfile), "sha256": _sha(mfile)}}
    rows = []
    tmp = os.path.join(out_dir, job_name(cell) + "_files")
    os.makedirs(tmp, exist_ok=True)
    for set_name in sets:
        files = _set_files(data_dir, set_name)
        if not files:
            continue
        ds = load_set(files)
        phys = _physics(cell.system, cell.material, ds)
        nc, ns = ds.n_cells, ds.X.shape[2]
        points = [load_meta(p)["point"] for p in files]
        by_arm = {}
        if cell.arm in ("C", "MUSCL-C"):
            for g in q.get("grids", [16, 32, 64, 128] if cell.arm == "C" else [32, 64]):
                paths = []
                for j, pt in enumerate(points):
                    p = os.path.join(tmp, f"{set_name}_{cell.arm}_{g}_{j}.npz")
                    res = run_c(cell.system, pt, cell.material, N_c=g, path=p, n_snap=ns, n_target=nc,
                                muscl=cell.arm == "MUSCL-C")
                    paths.append(p if res.ok else None)
                by_arm[f"{cell.arm}@{g}"] = paths
        else:
            from p3b.train.cplusl import rollout_cplusl_each
            by_arm["C+L"] = rollout_cplusl_each(mult, points, cell.material, q.get("n_coarse", 32), nc, ns, pilot["K"],
                                                [os.path.join(tmp, f"{set_name}_CL_{j}.npz") for j in range(len(points))])
        n_cycles = q.get("n_cycles", ds.X.shape[1])
        for r in evaluate_files(by_arm, ds, phys, n_cycles, cell.seed, pilot.get("floors"), pilot.get("tau_S")):
            r.update(set=set_name, design_index=int(points[r["trajectory"]]["index"]))
            rows.append(r)
    meta = {"arm": cell.arm, "system": cell.system, "material": cell.material, "size": cell.size,
            "seed": cell.seed, "pilot": pilot, "sets": list(sets), **meta_extra}
    suffix = "" if train else "_" + "_".join(sets)
    path = os.path.join(out_dir, job_name(cell) + suffix + ".json")
    write_results(path, rows, meta)
    return Path(path)
