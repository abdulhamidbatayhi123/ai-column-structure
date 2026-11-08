"""Provenance: every output file carries the git commit it was produced at."""
import os
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def git_commit() -> str:
    """HEAD hash, with "-dirty" if tracked files differ from it.

    Outside a git checkout (the Kaggle kit) it returns "kit:<hash>" from the kit's kit_commit.txt,
    and "unknown" only if neither exists.
    """
    kit = os.path.join(ROOT, "kit_commit.txt")
    if os.path.exists(kit) and not os.path.exists(os.path.join(ROOT, ".git")):
        with open(kit) as f:
            return "kit:" + f.read().strip()
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("-dirty" if dirty else "")
