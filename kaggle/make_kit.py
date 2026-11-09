"""Build the Kaggle dataset kit for reference-data generation (plan 1, Task 9).

The kit mirrors the repository layout that the code expects, so `p3b` finds its data relative to
its own root:
    p3b/  kaggle/generate_chunk.py  scripts/time_reference.py  verify/materials/fit_isotherms.json
    data/design/*.json  paper1/{isotherm,solver_fd,gen_parametric_dataset}.py
    kit_commit.txt  MANIFEST.json (SHA-256 of every file, Paper 1's source)  dataset-metadata.json

Paper 1's three files are downloaded read-only from the public repository at release v2.0.1 (the
same files CI checks out), unless `paper1_src` names a local directory (tests).

    python kaggle/make_kit.py OUT_DIR          # commit = current HEAD (must be clean)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p3b.design import chunk_indices  # noqa: E402,F401  (re-exported)

ROOT = Path(__file__).resolve().parents[1]
PAPER1_FILES = ("isotherm.py", "solver_fd.py", "gen_parametric_dataset.py")
PAPER1_REF = "v2.0.1"
PAPER1_RAW = "https://raw.githubusercontent.com/abdulhamidbatayhi123/ai-mof-cof-dynamics/{ref}/{name}"
KIT_SLUG = "abdulhamidbatayhi/p3b-reference-kit"


def _copy_tree(src: Path, dst: Path, pattern: str) -> None:
    for p in sorted(src.glob(pattern)):
        if p.is_file() and "__pycache__" not in p.parts:
            target = dst / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)


def _fetch_paper1(dst: Path) -> str:
    for name in PAPER1_FILES:
        req = urllib.request.Request(PAPER1_RAW.format(ref=PAPER1_REF, name=name),
                                     headers={"User-Agent": "p3b-kit-builder"})
        with urllib.request.urlopen(req, timeout=60) as r:
            (dst / name).write_bytes(r.read())
    return f"github:abdulhamidbatayhi123/ai-mof-cof-dynamics@{PAPER1_REF}"


def make_kit(out_dir, commit: str, paper1_src=None) -> Path:
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    _copy_tree(ROOT, out, "p3b/**/*.py")              # the whole package, subpackages included
    _copy_tree(ROOT, out, "kaggle/train_chunk.py")
    _copy_tree(ROOT, out, "scripts/pilot_rules.py")
    _copy_tree(ROOT, out, "kaggle/generate_chunk.py")
    _copy_tree(ROOT, out, "scripts/time_reference.py")
    _copy_tree(ROOT, out, "verify/v1_large_grid.py")
    _copy_tree(ROOT, out, "verify/materials/fit_isotherms.json")
    _copy_tree(ROOT, out, "data/design/*.json")
    # the pilot (Plan 5 Task 6): its runner, the chosen values of earlier phases, the phase-1
    # autoencoders and, when staged locally, the pilot data; absent files are simply not copied
    _copy_tree(ROOT, out, "kaggle/pilot_chunk.py")
    _copy_tree(ROOT, out, "results/pilot_stage.json")
    _copy_tree(ROOT, out, "results/pilot_ae/*.pt")
    _copy_tree(ROOT, out, "data/pilot/**/*.npz")
    p1 = out / "paper1"
    p1.mkdir()
    if paper1_src is None:
        source = _fetch_paper1(p1)
    else:
        for name in PAPER1_FILES:
            shutil.copy2(Path(paper1_src) / name, p1 / name)
        source = f"local:{Path(paper1_src).resolve()}"
    (out / "kit_commit.txt").write_text(commit + "\n")
    files = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(out.rglob("*")) if p.is_file()}
    (out / "MANIFEST.json").write_text(json.dumps({"commit": commit, "paper1_source": source, "files": files},
                                                  indent=1))
    (out / "dataset-metadata.json").write_text(json.dumps(
        {"title": "p3b-reference-kit", "id": KIT_SLUG, "licenses": [{"name": "other"}]}, indent=1))
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    from p3b.provenance import git_commit
    c = git_commit()
    if c.endswith("-dirty") or c == "unknown":
        sys.exit(f"refusing to build a kit from an unclean tree ({c})")
    print(make_kit(sys.argv[1], c))
