"""The study's cells, their Kaggle job names, the compute ceiling, and one cell's run
(PREREG Section 0 design table and Section 11; docs/compute_3B.md Section 3; Plan 4, Task 4).

A cell is (arm, system, material, training size, seed). U+proj and U+F+proj have no cells of their
own: they reuse U's and U+F's trained weights, so they are evaluated in the same run
(`evaluate_also`). All learned arms of one (system, material, size, seed) share one frozen
autoencoder: the first cell trains it and saves it, and every later cell loads it.
"""
from __future__ import annotations

import glob
import os
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


def job_name(c: Cell) -> str:
    return f"p3b-{_ARM[c.arm]}-{c.system.lower()}{_MAT[c.material]}-{c.size}-s{c.seed}"


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


def apply_reductions(cs: list, steps: list) -> list:
    return [c for c in cs if not any(_removed_by(s, c) for s in steps)]


def plan_reductions(spent_h: float, remaining: list, cost_h: dict, ceiling_h: float = 700.0) -> list:
    """The reduction steps, in order, needed so that spent + the projected cost of the remaining
    cells fits under the ceiling; all seven if even that does not fit (the never-reduced core then
    runs and the excess is reported)."""
    steps = []
    for step in [None] + REDUCTION_ORDER:
        if step is not None:
            steps.append(step)
        if spent_h + sum(cost_h[c] for c in apply_reductions(remaining, steps)) <= ceiling_h:
            return steps
    return steps


# ---- one cell's run ------------------------------------------------------------------------
def _physics(system, material, ds):
    from p3b.latent.physics import ColumnPhysics, StorePhysics
    if system == "A":
        from p3b.simulate_a import setup_a
        return ColumnPhysics(setup_a(ds.meta[0]["point"], material, ds.n_cells)[0])
    from p3b.simulate_b import T_AMB, make_store
    return StorePhysics(make_store(ds.n_cells), T_AMB)


def run_cell(cell: Cell, data_dir: str, out_dir: str, pilot: dict, quick: dict | None = None):
    """Train the cell's model (loading or training the shared autoencoder), evaluate it and its
    evaluate_also arms on the test set, and write out_dir/<job name>.json. `quick` shrinks sizes for
    tests only; the study passes None."""
    from p3b.eval.evaluate import evaluate, write_results
    from p3b.latent.arms_structured import SDynamics, SLDynamics
    from p3b.latent.arms_unstructured import UDynamics, UFDynamics
    from p3b.latent.autoencoder import MLPAutoencoder, train_autoencoder
    from p3b.latent.data import Standardiser, load_set
    from p3b.latent.fno import FNO1d
    from p3b.latent.functionals import LatentFunctionals
    from p3b.train.trainer import Arm, TermsStandardiser, TrainConfig, time_scale, train

    if cell.arm in ("C", "C+L", "MUSCL-C"):
        raise NotImplementedError("solver-arm cells run through p3b.solvers (Plan 4 Task 4b)")
    q = quick or {}
    os.makedirs(out_dir, exist_ok=True)
    ds = load_set(sorted(glob.glob(os.path.join(data_dir, "train", "*.npz"))))
    test = load_set(sorted(glob.glob(os.path.join(data_dir, "test", "*.npz"))))
    A = cell.system == "A"
    n_ch, n_out = (3, 2) if A else (2, 1)
    nc, n_state, k = ds.n_cells, ds.X.shape[-1], ds.features.shape[-1]
    std = Standardiser.fit(ds.X, nc, n_ch, n_out, features=ds.features)
    d = int(pilot["d"])
    torch.manual_seed(cell.seed)
    ae = MLPAutoencoder(n_state, d, width=q.get("ae_width", 256))
    ae_path = os.path.join(out_dir, f"ae_{cell.system}_{cell.material}_{cell.size}_s{cell.seed}.pt")
    if os.path.exists(ae_path):
        ae.load_state_dict(torch.load(ae_path))
    else:
        train_autoencoder(ae, std.encode(ds.X), epochs=q.get("ae_epochs", 60), seed=cell.seed)
        torch.save(ae.state_dict(), ae_path)
    phys = _physics(cell.system, cell.material, ds)
    lf = LatentFunctionals(ae.decode, std, phys, cell.system, nc)
    ts = time_scale(ds)
    width = q.get("width", pilot.get("width", {}).get(cell.arm, 128))
    kind = "U" if cell.arm == "U-long" else cell.arm
    terms = None
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
    arm = Arm(kind, dyn, None if kind == "FNO" else ae, std, None if kind == "FNO" else lf, terms=terms)
    iters = pilot["iterations"] if cell.arm != "U-long" else pilot.get("iterations_ulong", pilot["iterations"])
    hist = train(arm, ds, TrainConfig(lr=pilot["lr"][cell.arm if cell.arm in pilot["lr"] else kind],
                                      iterations=iters, seed=cell.seed, h_steps=pilot["h_steps"],
                                      n_windows=q.get("n_windows", 8), n_steps=q.get("n_steps", 20)))
    n_cycles = q.get("n_cycles", test.X.shape[1])
    rows = []
    for name in (cell.arm,) + tuple(cell.evaluate_also):
        ev_kind = {"U+proj": "U+proj", "U+F+proj": "U+F+proj"}.get(name, kind)
        ev = Arm(ev_kind, dyn, arm.ae, std, arm.lf, terms=terms)
        for r in evaluate(ev, test, phys, n_cycles, pilot["h_steps"], pilot["floors"], pilot["tau_S"], cell.seed):
            rows.append(dict(r, arm=name))
    meta = {"arm": cell.arm, "system": cell.system, "material": cell.material, "size": cell.size,
            "seed": cell.seed, "params": hist["params"], "pilot": pilot, "train_loss_last": hist["loss"][-1],
            "train_skipped": hist["skipped"], "train_cpu_s": hist["wall_s"], "flags": hist["flags"]}
    path = os.path.join(out_dir, job_name(cell) + ".json")
    write_results(path, rows, meta)
    from pathlib import Path
    return Path(path)
