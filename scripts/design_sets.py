"""Write every pre-registered design set to data/design/<system>_<set>.json.

The seeds are fixed here, once, as literals; the sets are generated once and never regenerated
with other seeds (PREREG_P3B_draft.md, Section 4). Each file records the seed and the git commit
it was produced at, with the Python, NumPy and SciPy versions. Existing files are never
overwritten unless --force is given (tests/test_cs_t6_fixes.py checks that every committed file
reproduces from its recorded seed).

    python scripts/design_sets.py [--force]
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SEEDS: dict[tuple[str, str], int] = {
    ("A_primary", "pilot_train"): 310_101,
    ("A_primary", "pilot_val"): 310_102,
    ("A_primary", "train"): 310_103,
    ("A_primary", "validation"): 310_104,
    ("A_primary", "long_validation"): 310_105,
    ("A_primary", "long_test"): 310_106,
    ("A_primary", "ood"): 310_107,
    ("A_primary", "scenarios"): 310_108,
    ("A_replication", "pilot"): 320_101,
    ("A_replication", "train"): 320_102,
    ("A_replication", "validation"): 320_103,
    ("A_replication", "long_validation"): 320_104,
    ("A_replication", "long_test"): 320_105,
    ("B", "pilot"): 330_101,
    ("B", "train"): 330_102,
    ("B", "validation"): 330_103,
    ("B", "long_validation"): 330_104,
    ("B", "long_test"): 330_105,
}


def main(out_dir: str = os.path.join(ROOT, "data", "design"), force: bool = False) -> None:
    import platform

    import numpy
    import scipy
    sys.path.insert(0, ROOT)
    from p3b.design import BOXES, make_set
    from p3b.provenance import git_commit

    os.makedirs(out_dir, exist_ok=True)
    paths = {key: os.path.join(out_dir, f"{key[0]}_{key[1]}.json") for key in SEEDS}
    existing = [p for p in paths.values() if os.path.exists(p)]
    if existing and not force:
        raise FileExistsError(f"{len(existing)} design files exist in {out_dir}; they are never "
                              "regenerated silently (use --force only to add version records)")
    commit = git_commit()
    versions = {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__}
    for (system, name), seed in SEEDS.items():
        pts = make_set(system, name, seed)
        box = "B" if system == "B" else ("A_OOD" if name == "ood" else "A")
        rec = {"system": system, "set": name, "seed": seed, "box": BOXES[box], "commit": commit,
               "sampler": "scipy.stats.qmc.Sobol(scramble=True).random_base2, first count points",
               "versions": versions, "points": pts}
        with open(paths[(system, name)], "w") as f:
            json.dump(rec, f, indent=1)
        print(f"{system:14s} {name:16s} {len(pts):4d} points, seed {seed}")


if __name__ == "__main__":
    main(force="--force" in sys.argv[1:])
