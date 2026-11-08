"""Locate Paper 1's public solver (ai-mof-cof-dynamics, release v2.0.1) and check its version.

Paper 1's code is not a package. It is checked out next to this repository, as in CI, or the
directory is named by the environment variable P3B_PAPER1_DIR. Importing this module checks the
three Paper 1 files 3B uses against their SHA-256 at release v2.0.1 (commit d1c7f66) and puts the
directory on sys.path, once. A different version raises ImportError instead of running silently.
"""
import hashlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
PAPER1_DIR = os.environ.get(
    "P3B_PAPER1_DIR", os.path.abspath(os.path.join(_HERE, "..", "..", "ai-mof-cof-dynamics")))

PAPER1_REF = "v2.0.1"
PAPER1_SHA256 = {
    "isotherm.py": "902f3ee00363464f03800cf17679e6636eca9febcea252a68b0aae4e562a4867",
    "solver_fd.py": "ed6dea1012915d3efd66412032cda344a42396bf56437079c6b4a212005a4da4",
    "gen_parametric_dataset.py": "3a8ee5cb9e02c890cf8f8807c27538f8707b0014bf9571b383606a3c2ccc941d",
}


def paper1_hashes(directory: str) -> dict:
    """SHA-256 of the three Paper 1 files in `directory` (None for a missing file)."""
    out = {}
    for name in PAPER1_SHA256:
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            out[name] = None
            continue
        with open(path, "rb") as f:
            out[name] = hashlib.sha256(f.read()).hexdigest()
    return out


def check_paper1(directory: str) -> None:
    """Raise ImportError unless every file in `directory` matches release v2.0.1."""
    found = paper1_hashes(directory)
    bad = [name for name, h in found.items() if h != PAPER1_SHA256[name]]
    if bad:
        raise ImportError(f"Paper 1 in {directory} is not release {PAPER1_REF}: "
                          f"{', '.join(bad)} missing or changed; check out ai-mof-cof-dynamics "
                          f"{PAPER1_REF} beside this repository or set P3B_PAPER1_DIR")


check_paper1(PAPER1_DIR)
if PAPER1_DIR not in sys.path:
    sys.path.append(PAPER1_DIR)
