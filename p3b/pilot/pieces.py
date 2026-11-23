"""The pilot's Kaggle pieces (Plan 5 Task 6): one function per job of kaggle/pilot_jobs.py. Each reads
the pilot data under data_root/<key>/{train,validation} and the chosen values of the earlier phases
(stage: p3b.pilot.collect.stage), and writes one piece file into out. Pieces with losses or errors
are read only by p3b.pilot.collect, which writes chosen values only; nothing here prints a loss.

Every piece file is {"meta": provenance, "value": ...}. The meta holds the SHA-256 of the stage the
piece ran under, so the collector never mixes pieces of different stages (plan 5 review I6), and
the kit commit, the SHA-256 of every data file and autoencoder used, q, the seeds and the torch
version (PREREG 10: reproducibility).

q holds overrides for tests on tiny data (grids, iterations, widths); the study runs with q = {}.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os

import numpy as np
import torch

from p3b.pilot import runs
from p3b.latent.data import layout  # noqa: E402
from p3b.pilot.rules import pilot_key

KEYS = {k: v for k, v in ((pilot_key(*sm), sm) for sm in (("A", "MOF-303"), ("A", "Al-fumarate"), ("B", None)))}
PRIMARY_KEY = pilot_key("A", "MOF-303")
LATENT_ARMS = ("U", "U+F", "P", "S", "S-L", "S-M", "S-Pi", "FNO")
LR_ARMS = LATENT_ARMS + ("C+L",)
CURVE_MAX, CURVE_EVERY = 16_000, 1_000          # PREREG Section 10: every 1,000 iterations up to 16,000
LR_ITERATIONS = 2_000
TWO_SEEDS = (0, 1)
STRUCTURED = ("S", "S-L", "S-M", "S-Pi")
# the stage keys each phase adds (p3b.pilot.collect.stage); a phase-p piece runs under the keys of phases < p
PHASE_KEYS = (("d", "decoder", "K", "tau_S"), ("lr",), ("iterations",), ("h_steps",))
# The revision of what the pieces compute (T8 RS-4, R1-4): it enters every stage hash, so every pilot kernel
# name and every piece's recorded stage change with it, and outputs of an earlier revision are never
# reused or collected. "r3": after R1 (the autoencoder's balance loss) and R3 (the regenerated pilot data).
# Bump it whenever a revision changes what a piece computes.
PILOT_REVISION = "r4"
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ---- stage and provenance ------------------------------------------------------------------------------
def restrict(stage: dict, phase: int) -> dict:
    """The stage a phase-`phase` piece runs under: the keys of phases before it."""
    keep = {k for p in PHASE_KEYS[:phase] for k in p}
    return {k: v for k, v in (stage or {}).items() if k in keep}


def stage_sha(stage: dict) -> str:
    """SHA-256 of the stage's chosen values (the phase keys only; runtime settings such as
    ae_input_root do not change what a piece computes) and of PILOT_REVISION."""
    chosen = {k: v for k, v in (stage or {}).items() if any(k in p for p in PHASE_KEYS)}
    chosen["_revision"] = PILOT_REVISION
    return hashlib.sha256(json.dumps(chosen, sort_keys=True).encode()).hexdigest()


def for_phase(stage: dict, phase: int) -> dict:
    """The stage handed to a phase-`phase` piece: the chosen values of earlier phases only (restrict),
    with any runtime settings (ae_input_root) kept; a phase-0 piece gets no chosen values whatever stage
    file the kit carries, so it records stage_sha({}) (T8 fix review, R1-4)."""
    phase_keys = {k for p in PHASE_KEYS for k in p}
    runtime = {k: v for k, v in (stage or {}).items() if k not in phase_keys and k != "_revision"}
    return {**runtime, **restrict(stage, phase)}


def load_stage(path, phase: int, required: bool = False) -> dict:
    """The stage file at `path` for a phase-`phase` job. Phase 0 runs under no chosen values, so its
    file is not read. For later phases the file must carry the current PILOT_REVISION (written by
    p3b.pilot.collect.stage): a stage of an earlier revision (for example the pre-R1/pre-R3 d and tau_S)
    is refused, never run under. A missing file gives {} unless `required`."""
    if phase == 0:
        return {}
    if not os.path.exists(str(path)):
        if required:
            raise FileNotFoundError(f"no stage file {path}: phase {phase} runs under the stage collected after "
                                    f"phase {phase - 1} (scripts/pilot_collect.py stage {phase - 1})")
        return {}
    with open(str(path)) as f:
        stage = json.load(f)
    if stage.get("_revision") != PILOT_REVISION:
        raise RuntimeError(f"stage file {path} is of pilot revision {stage.get('_revision')!r}, not "
                           f"{PILOT_REVISION!r}: recollect it (scripts/pilot_collect.py stage N) from pieces of "
                           f"the current revision")
    return stage


def _sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _data_used(*file_lists) -> dict:
    return {os.path.basename(p): _sha_file(p) for fl in file_lists for p in fl}


def _kit_commit() -> str | None:
    p = os.path.join(_ROOT, "kit_commit.txt")
    if os.path.exists(p):
        return open(p).read().strip()
    try:
        import subprocess
        return subprocess.run(["git", "-C", _ROOT, "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:                                     # no git: recorded as unknown, never invented
        return None


def _write(out, name: str, value, stage: dict | None = None, q: dict | None = None, used: dict | None = None,
           seeds=None) -> str:
    os.makedirs(str(out), exist_ok=True)
    path = os.path.join(str(out), name)
    meta = {"stage_sha256": stage_sha(stage), "kit_commit": _kit_commit(), "q": q or {}, "used": used or {},
            "seeds": list(seeds) if seeds is not None else None, "torch": torch.__version__}
    tmp = path + ".partial"
    with open(tmp, "w") as f:
        json.dump({"meta": meta, "value": value}, f)
    os.replace(tmp, path)
    return path


# ---- data ------------------------------------------------------------------------------------------------
def files(data_root, key: str, split: str) -> list:
    from p3b.gates import load_meta
    fs = glob.glob(os.path.join(str(data_root), key, split, "*.npz"))
    return sorted(fs, key=lambda p: int(load_meta(p)["point"]["index"]))


def long_files(data_root, key: str) -> list:
    """Every trajectory of the key with the key's full cycle count (CSS and reference noise need the
    long ones: system A primary's pilot_val; all 8 of A_replication and B)."""
    fs = files(data_root, key, "train") + files(data_root, key, "validation")
    n = {p: int(np.load(p)["ports_cum"].shape[0]) for p in fs}
    top = max(n.values())
    return [p for p in fs if n[p] == top]


# ---- phase 0 -------------------------------------------------------------------------------------------
def piece_data(data_root, key: str, out, q: dict | None = None, stage: dict | None = None) -> str:
    """CSS per long trajectory, the decoder gate over the d grid, the reference noise."""
    q = q or {}
    lf, tr, va = long_files(data_root, key), files(data_root, key, "train"), files(data_root, key, "validation")
    grid = q.get("d_grid", (16, 24, 32, 48, 64))
    gate = runs.decoder_gate_grid(tr, va, grid, q.get("ae_epochs", 60), q.get("ae_width", 256))
    value = {"css": runs.css_cycles(lf), "decoder_gate": {str(k): v for k, v in gate.items()},
             "reference_noise": runs.reference_noise(lf)}
    if not any(g["passed"] for g in gate.values()):     # PREREG 4: the convolutional decoder on the same grid
        conv = runs.decoder_gate_grid(tr, va, grid, q.get("ae_epochs", 60), q.get("ae_width", 256), kind="conv")
        value["decoder_gate_conv"] = {str(k): v for k, v in conv.items()}
    return _write(out, f"data_{key}.json", value, stage, q, {"data": _data_used(tr, va)}, seeds=[0])


def piece_cplk(data_root, key: str, out, q: dict | None = None, stage: dict | None = None) -> str:
    q = q or {}
    sys_, mat = KEYS[key]
    va = files(data_root, key, "validation")
    eps = runs.cplusl_K(va, mat, n_coarse=q.get("n_coarse", 32), Ks=q.get("Ks", (1, 2, 4, 8)),
                     workers=q.get("workers", min(4, os.cpu_count() or 1)))
    return _write(out, f"cplusl_K_{key}.json", {str(k): v for k, v in eps.items()}, stage, q,
                  {"data": _data_used(va)})


# ---- arms ------------------------------------------------------------------------------------------------
def _cell(arm: str, key: str, seed: int):
    from p3b.train.cells import Cell
    sys_, mat = KEYS[key]
    return Cell(arm, sys_, mat, "pilot", seed)


def _std(ds):
    from p3b.latent.data import Standardiser
    n_ch, n_out = layout(ds.system)
    return Standardiser.fit(ds.X, ds.n_cells, n_ch, n_out, features=ds.features)


def piece_ae(data_root, key: str, seed: int, stage: dict, out, q: dict | None = None) -> str:
    """The pilot autoencoder of (key, seed) at the chosen d, saved under the study's naming
    (ae_<system>_<material>_pilot_s<seed>.pt) so p3b.train.cells._build finds it; its SHA-256 is
    recorded by every piece that uses it."""
    from p3b.latent.autoencoder import fit_autoencoder
    from p3b.latent.data import load_set
    q = q or {}
    ds = load_set(files(data_root, key, "train"))
    std = _std(ds)
    torch.manual_seed(seed)
    ae = _ae(ds, stage, key, q)
    fit_autoencoder(ae, ds, std, runs._physics(ds), epochs=q.get("ae_epochs", 60), seed=seed,
                    **({"weight": q["ae_balance_weight"]} if "ae_balance_weight" in q else {}))   # tiny-data tests only
    c = _cell("U", key, seed)
    os.makedirs(str(out), exist_ok=True)
    path = os.path.join(str(out), f"ae_{c.system}_{c.material}_{c.size}_s{seed}.pt")
    torch.save(ae.state_dict(), path)
    return path


def _decoder(stage, key) -> str:
    return (stage.get("decoder") or {}).get(key, "mlp")


def _ae(ds, stage, key, q):
    from p3b.latent.autoencoder import make_autoencoder
    n_ch, n_out = layout(ds.system)
    return make_autoencoder(_decoder(stage, key), ds.n_cells, n_ch, n_out, int(stage["d"][key]),
                            width=q.get("ae_width", 256))


def _pilot(stage, key, kappa):
    return {"d": stage["d"][key], "decoder": _decoder(stage, key), "kappa": kappa,
            "ae_input_root": stage.get("ae_input_root", "/kaggle/input")}


def _kappa(key, seed, stage, ds, out, q, used=None):
    from p3b.latent.functionals import LatentFunctionals
    from p3b.train.cells import _load_ae, _physics
    c = _cell("S", key, seed)
    std = _std(ds)
    ae = _ae(ds, stage, key, q)
    meta = _load_ae(c, ae, str(out), _pilot(stage, key, None), {**q, "require_ae": True}, std, ds)
    if used is not None:
        used.setdefault("autoencoders", {})[meta["file"]] = meta["sha256"]
    lf = LatentFunctionals(ae.decode, std, _physics(c.system, c.material, ds), c.system, ds.n_cells)
    return runs.kappa_from_data(lf, ae, std, ds), runs.g2_from_data(lf, ae, std, ds)


def make_arm(arm: str, key: str, seed: int, stage: dict, ds, out, q: dict | None = None, used: dict | None = None):
    """A fresh pilot arm built exactly as the study builds its cells (p3b.train.cells._build): matched
    widths, the shared autoencoder of (key, seed), S's kappa from the encoded pilot data and seed 0's
    autoencoder (one kappa per key, as in the study). The
    autoencoder's SHA-256 goes into `used`."""
    from p3b.train.cells import _build
    q = q or {}
    # one kappa per key, from seed 0's autoencoder, for every seed: the study's cells use pilot_values'
    # single kappa per key (p3b.train.cells), so the pilot's two-seed runs must too (M7)
    kappa = _kappa(key, 0, stage, ds, out, q, used)[0] if arm in STRUCTURED else None
    built, _, _, meta = _build(_cell(arm, key, seed), _pilot(stage, key, kappa), {**q, "require_ae": True}, ds,
                               str(out))
    if used is not None and meta.get("autoencoder"):
        used.setdefault("autoencoders", {})[meta["autoencoder"]["file"]] = meta["autoencoder"]["sha256"]
    return built


def _primary(data_root):
    from p3b.latent.data import load_set
    tr, va = files(data_root, PRIMARY_KEY, "train"), files(data_root, PRIMARY_KEY, "validation")
    return load_set(tr), load_set(va), {"data": _data_used(tr, va)}


# ---- phase 1 and 2 ---------------------------------------------------------------------------------------
def _fingerprint(piece: str, stage: dict, q: dict, used: dict) -> dict:
    """What every session of a resumable piece must share (T8 RS-5): the piece, the stage it runs under
    (with PILOT_REVISION), the kit commit, the overrides q and the data files' SHA-256."""
    from p3b.resume import plain
    return plain({"piece": piece, "stage_sha256": stage_sha(stage), "kit_commit": _kit_commit(), "q": q or {},
                  "data": (used or {}).get("data", {})})


def _session_in(resume_from, out, prefix: str, fingerprint: dict):
    """A resumed piece's state (None for a first session): its <prefix>_state.json and <prefix>*_ckpt.pt
    files are copied from the earlier session's output (searched recursively, as Kaggle mounts it),
    after the RESUME.json beside them showed the same run fingerprint."""
    import shutil

    from p3b.resume import check_resume
    os.makedirs(str(out), exist_ok=True)
    if not resume_from:
        return None
    states = glob.glob(os.path.join(str(resume_from), "**", f"{prefix}_state.json"), recursive=True)
    if len(states) != 1:
        raise FileNotFoundError(f"{len(states)} {prefix}_state.json under {resume_from} (expected one)")
    check_resume(os.path.dirname(states[0]), fingerprint)
    for p in glob.glob(os.path.join(os.path.dirname(states[0]), f"{prefix}*_ckpt.pt")):
        shutil.copy(p, os.path.join(str(out), os.path.basename(p)))
    with open(states[0]) as f:
        return json.load(f)


def _prev_autoencoders(resume_from, prefix: str):
    """The autoencoders ({file: SHA-256}) the earlier session of a resumed piece recorded in the
    RESUME.json beside its <prefix>_state.json; None for a first session."""
    if not resume_from:
        return None
    states = glob.glob(os.path.join(str(resume_from), "**", f"{prefix}_state.json"), recursive=True)
    if len(states) != 1:
        raise FileNotFoundError(f"{len(states)} {prefix}_state.json under {resume_from} (expected one)")
    with open(os.path.join(os.path.dirname(states[0]), "RESUME.json")) as f:
        return json.load(f).get("autoencoders") or {}


def _same_autoencoders(prev, used: dict) -> None:
    """A continuation must use the earlier sessions' autoencoders (T8 fix review, RS-5): the file found
    under the inputs is re-read every session, so a rebuilt kit or a re-pushed autoencoder job would
    otherwise resume the checkpoint under another encoder."""
    if prev is None:
        return
    now = (used or {}).get("autoencoders", {})
    diff = sorted(f for f in set(prev) & set(now) if prev[f] != now[f])
    if diff:
        raise RuntimeError(f"autoencoder {diff[0]} has SHA-256 {now[diff[0]][:12]}, the earlier session used "
                           f"{prev[diff[0]][:12]}: the run fingerprint differs; start the job afresh")


def _arm_checked(prev, arm: str, key: str, seed: int, stage: dict, ds, out, q, used: dict):
    """make_arm, then the check that its autoencoder is the earlier sessions' one."""
    built = make_arm(arm, key, seed, stage, ds, out, q, used)
    _same_autoencoders(prev, used)
    return built


def _resume_json(out, piece: str, stopped_at: int, fingerprint: dict, progress: list, **extra) -> str:
    """RESUME.json of a session that ran out of time (run_queue continues the job). `progress` must grow
    from session to session: run_queue stops a chain whose progress does not (T8 RS-1)."""
    path = os.path.join(str(out), "RESUME.json")
    with open(path, "w") as f:
        json.dump({"piece": piece, "stopped_at": stopped_at, "progress": progress, **extra,
                   "stage_sha256": fingerprint["stage_sha256"], "fingerprint": fingerprint}, f, indent=1)
    return path


def _session_out(out, prefix: str, piece: str, res: dict, fingerprint: dict, prev_aes=None,
                 used: dict | None = None) -> str:
    """A session that ran out of time: its state and RESUME.json. A sweep's progress is (rates finished,
    iteration of the running rate), since the iteration restarts with every rate. The autoencoders of
    this and the earlier sessions are recorded for the next session's check."""
    with open(os.path.join(str(out), f"{prefix}_state.json"), "w") as f:
        json.dump(res["state"], f)
    st = res["state"]
    progress = ([len(st["done"])] if isinstance(st, dict) and "done" in st else []) + [int(res["stopped_at"])]
    aes = {**(prev_aes or {}), **((used or {}).get("autoencoders") or {})}
    return _resume_json(out, piece, res["stopped_at"], fingerprint, progress, state=f"{prefix}_state.json",
                        autoencoders=aes)


def piece_lr(arm: str, data_root, stage: dict, out, q: dict | None = None, deadline: float | None = None,
             resume_from: str | None = None) -> str:
    """The learning-rate sweep; it spans Kaggle sessions (RESUME.json past `deadline`, continued from
    `resume_from`), the result equal to one done in a single session."""
    q = q or {}
    ds, val, used = _primary(data_root)
    rates = q.get("lr_grid", (3e-4, 1e-3, 3e-3))
    its = q.get("lr_iterations", LR_ITERATIONS)
    nw, ns = q.get("n_windows", 8), q.get("n_steps", 20)
    prefix = f"lr_{_tag(arm)}"
    fp = _fingerprint(f"lr-{_tag(arm)}", stage, q, used)
    state = _session_in(resume_from, out, prefix, fp)
    prev = _prev_autoencoders(resume_from, prefix)
    if arm == "C+L":
        res = runs.cplusl_lr_sweep(ds, val, KEYS[PRIMARY_KEY][1], int(stage["K"][PRIMARY_KEY]), rates, its,
                                   n_coarse=q.get("n_coarse", 32), n_windows=nw, n_steps=ns,
                                   checkpoint_dir=str(out), deadline=deadline, state=state)
    else:                       # checkpoints lr_<tag>_<lr>_ckpt.pt, which _session_in copies (T8 RS-1)
        res = runs.lr_sweep(lambda: _arm_checked(prev, arm, PRIMARY_KEY, 0, stage, ds, out, q, used), ds, val, rates,
                            its, h_steps=1, seed=0, n_windows=nw, n_steps=ns, checkpoint_dir=str(out),
                            deadline=deadline, state=state, tag=_tag(arm))
    if "stopped_at" in res:
        return _session_out(out, prefix, f"lr-{_tag(arm)}", res, fp, prev, used)
    return _write(out, f"lr_{_tag(arm)}.json", {str(k): v for k, v in res.items()}, stage, q, used, seeds=[0])


def _resume_into(resume_from, out, name: str, fingerprint: dict) -> None:
    """Copy an earlier session's file `name` (searched recursively, as Kaggle mounts kernel outputs; exactly
    one copy), after the RESUME.json beside it showed the same run fingerprint."""
    import shutil

    from p3b.resume import check_resume
    hits = glob.glob(os.path.join(str(resume_from), "**", name), recursive=True)
    if len(hits) != 1:
        raise FileNotFoundError(f"{len(hits)} {name} under {resume_from} (expected one)")
    check_resume(os.path.dirname(hits[0]), fingerprint)
    os.makedirs(str(out), exist_ok=True)
    shutil.copy(hits[0], os.path.join(str(out), name))


def piece_curve(arm: str, data_root, stage: dict, out, q: dict | None = None, deadline: float | None = None,
                resume_from: str | None = None) -> str:
    """The arm's curve to CURVE_MAX iterations. Past `deadline` it checkpoints and writes RESUME.json
    instead of the curve, and a later session continues from `resume_from` (the earlier session's
    output). C+L's run spans sessions: at 8.1 s per iteration (K = 4, results/timing_cplusl_train.json)
    16,000 iterations take about 36 h, four Kaggle sessions."""
    q = q or {}
    ds, val, used = _primary(data_root)
    its, every = q.get("curve_iterations", CURVE_MAX), q.get("curve_every", CURVE_EVERY)
    nw, ns = q.get("n_windows", 8), q.get("n_steps", 20)
    lr = stage["lr"][arm]
    fp = _fingerprint(f"curve-{_tag(arm)}", stage, q, used)
    if arm == "C+L":
        ckpt = f"curve_{_tag(arm)}_ckpt.pt"
        os.makedirs(str(out), exist_ok=True)
        if resume_from:
            _resume_into(resume_from, out, ckpt, fp)
        curve = runs.cplusl_curve(ds, val, KEYS[PRIMARY_KEY][1], lr, its, every, int(stage["K"][PRIMARY_KEY]), 0,
                                  n_coarse=q.get("n_coarse", 32), n_windows=nw, n_steps=ns,
                                  checkpoint=os.path.join(str(out), ckpt), deadline=deadline)
        if isinstance(curve, dict):                      # the session ends; a later one resumes
            return _resume_json(out, f"curve-{_tag(arm)}", curve["stopped_at"], fp, [int(curve["stopped_at"])],
                                iterations=its, checkpoint=ckpt)
    else:
        prefix = f"curve_{_tag(arm)}"
        state = _session_in(resume_from, out, prefix, fp)
        prev = _prev_autoencoders(resume_from, prefix)
        curve = runs.train_curve(_arm_checked(prev, arm, PRIMARY_KEY, 0, stage, ds, out, q, used), ds, val, lr, its,
                                 every, 1, 0, nw, ns, checkpoint=os.path.join(str(out), f"{prefix}_ckpt.pt"),
                                 deadline=deadline, state=state)
        if isinstance(curve, dict):
            return _session_out(out, prefix, f"curve-{_tag(arm)}", curve, fp, prev, used)
    return _write(out, f"curve_{_tag(arm)}.json", [list(x) for x in curve], stage, q, used, seeds=[0])


def _tag(arm: str) -> str:
    return arm.replace("+", "p").replace("-", "")


# ---- phase 3 ---------------------------------------------------------------------------------------------
def _train_in_sessions(out, prefix: str, resume_from, fingerprint: dict):
    """The checkpoint path of a piece that trains one run across sessions (T8 RS-8): the earlier
    session's state and checkpoint are copied in, and the checkpoint must hold the state's iteration."""
    state = _session_in(resume_from, out, prefix, fingerprint)
    ckpt = os.path.join(str(out), f"{prefix}_ckpt.pt")
    if state is not None:
        runs._check_checkpoint(ckpt, int(state["it"]))
    return ckpt


def piece_s(key: str, data_root, stage: dict, out, q: dict | None = None, deadline: float | None = None,
            resume_from: str | None = None) -> str:
    """S trained at its pilot lr and iterations with one RK4 step per snapshot interval on the key's
    pilot training data; the h check on its validation data with the reference's tau_S; the raw
    |P grad S|^2 values and the projection defects per step and h (threshold inputs). An S whose
    training skipped more than 5 % of its iterations is not checked: the piece stops (review I5).
    Training spans sessions like the sweeps: past `deadline` it checkpoints and writes RESUME.json, and
    a later session continues from `resume_from` (T8 RS-8)."""
    from p3b.latent.data import load_set
    from p3b.train import trainer
    q = q or {}
    tr, va = files(data_root, key, "train"), files(data_root, key, "validation")
    ds, val = load_set(tr), load_set(va)
    used = {"data": _data_used(tr, va)}
    prefix = f"s_{key}"
    fp = _fingerprint(f"s-{key}", stage, q, used)
    ckpt = _train_in_sessions(out, prefix, resume_from, fp)
    prev = _prev_autoencoders(resume_from, prefix)
    arm = _arm_checked(prev, "S", key, 0, stage, ds, out, q, used)
    its = int(q.get("iterations", stage["iterations"]))
    hist = trainer.train(arm, ds, trainer.TrainConfig(lr=stage["lr"]["S"], iterations=its, seed=0, h_steps=1,
                                                      n_windows=q.get("n_windows", 8), n_steps=q.get("n_steps", 20),
                                                      checkpoint=ckpt, deadline=deadline))
    if "stopped_at" in hist:                             # the session ends; a later one resumes
        it = int(hist["stopped_at"])
        return _session_out(out, prefix, f"s-{key}", {"stopped_at": it, "state": {"it": it}}, fp, prev, used)
    if runs.too_many_skipped(hist, its):
        raise RuntimeError(f"S on {key}: more than 5 % of its training iterations were skipped; no h check")
    hc = runs.h_check(arm, val, float(stage["tau_S"][key]), steps=q.get("h_steps_grid", (1, 2, 4)))
    g2 = _kappa(key, 0, stage, ds, out, q, used)[1]
    value = {"h_check": {str(k): v for k, v in hc["h"].items()}, "g2": g2,
             "defects_per_step": {str(k): v for k, v in hc["defects_per_step"].items()},
             "train": {"iterations": its, "skipped": hist.get("skipped", 0), "skipped_grad": hist.get("skipped_grad", 0)}}
    return _write(out, f"s_{key}.json", value, stage, q, used, seeds=[0])


def piece_wall(data_root, stage: dict, out, q: dict | None = None) -> str:
    """Training seconds per iteration of every latent arm (PREREG 10: the cost of every arm; U and S
    set U-long's iteration count)."""
    from p3b.latent.data import load_set
    q = q or {}
    tr = files(data_root, PRIMARY_KEY, "train")
    ds = load_set(tr)
    used = {"data": _data_used(tr)}
    makers = {a: (lambda a=a: make_arm(a, PRIMARY_KEY, 0, stage, ds, out, q, used)) for a in LATENT_ARMS}
    kw = dict(iterations=q.get("wall_iterations", 50), warmup=q.get("wall_warmup", 5),
              n_windows=q.get("n_windows", 8), n_steps=q.get("n_steps", 20))
    res = runs.wall_per_iter(makers, ds, **kw)                  # at the session's threads (the rule input)
    threads = torch.get_num_threads()
    try:                                                        # one thread: whether four 1-thread jobs per
        torch.set_num_threads(1)                                # session beat one 4-thread job (job layout)
        used["timing_1_thread"] = runs.wall_per_iter(makers, ds, **kw)
    finally:
        torch.set_num_threads(threads)
    return _write(out, "wall_per_iter.json", res, stage, q, used, seeds=[0])


# ---- phase 4 ---------------------------------------------------------------------------------------------
def piece_two(arm: str, seed: int, data_root, stage: dict, out, q: dict | None = None, quarter: bool = False,
              deadline: float | None = None, resume_from: str | None = None) -> str:
    """One seed of U or S (or U at a quarter of the iterations: the positive control) on system A
    primary's pilot data at the chosen lr, iterations and h; evaluation rows on pilot_val and the
    run's training counts. Training spans sessions like the sweeps (T8 RS-8)."""
    q = q or {}
    ds, val, used = _primary(data_root)
    its = int(q.get("iterations", stage["iterations"]))
    its = max(1, its // 4) if quarter else its
    prefix = f"rows_{'Uq' if quarter else _tag(arm)}_s{seed}"
    piece = f"pcq-s{seed}" if quarter else f"two-{_tag(arm)}-s{seed}"
    fp = _fingerprint(piece, stage, q, used)
    ckpt = _train_in_sessions(out, prefix, resume_from, fp)
    prev = _prev_autoencoders(resume_from, prefix)
    res = runs.two_seed_rows(lambda: _arm_checked(prev, arm, PRIMARY_KEY, seed, stage, ds, out, q, used), ds, val,
                             stage["lr"][arm], its, int(stage["h_steps"][PRIMARY_KEY]), seeds=(seed,),
                             n_windows=q.get("n_windows", 8), n_steps=q.get("n_steps", 20), checkpoint=ckpt,
                             deadline=deadline)
    if isinstance(res, dict):                            # the session ends; a later one resumes
        it = int(res["stopped_at"])
        return _session_out(out, prefix, piece, {"stopped_at": it, "state": {"it": it}}, fp, prev, used)
    rows, train = res
    return _write(out, f"{prefix}.json", {"rows": rows[seed], "train": train[seed]}, stage, q, used, seeds=[seed])
