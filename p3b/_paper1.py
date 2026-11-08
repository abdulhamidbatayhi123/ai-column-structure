"""Locate Paper 1's public solver (ai-mof-cof-dynamics, release v2.0.1).

Paper 1's code is not a package. It is checked out next to this repository, as in CI, or the
directory is named by the environment variable P3B_PAPER1_DIR. Importing this module puts that
directory on sys.path, once.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
PAPER1_DIR = os.environ.get(
    "P3B_PAPER1_DIR", os.path.abspath(os.path.join(_HERE, "..", "..", "ai-mof-cof-dynamics")))

if not os.path.isfile(os.path.join(PAPER1_DIR, "solver_fd.py")):
    raise ImportError(f"Paper 1's solver_fd.py not found in {PAPER1_DIR}; "
                      "check out ai-mof-cof-dynamics v2.0.1 beside this repository or set P3B_PAPER1_DIR")
if PAPER1_DIR not in sys.path:
    sys.path.append(PAPER1_DIR)
