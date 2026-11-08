"""Write every pre-registered design set to data/design/<system>_<set>.json.

The seeds are fixed here, once, as literals; the sets are generated once and never regenerated
with other seeds (PREREG_P3B_draft.md, Section 4). Each file records the seed and the git commit
it was produced at.

    python scripts/design_sets.py
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


def main(out_dir: str = os.path.join(ROOT, "data", "design")) -> None:
    sys.path.insert(0, ROOT)
    from p3b.design import BOXES, make_set
    from p3b.provenance import git_commit

    os.makedirs(out_dir, exist_ok=True)
    commit = git_commit()
    for (system, name), seed in SEEDS.items():
        pts = make_set(system, name, seed)
        box = "B" if system == "B" else ("A_OOD" if name == "ood" else "A")
        rec = {"system": system, "set": name, "seed": seed, "box": BOXES[box], "commit": commit,
               "sampler": "scipy.stats.qmc.Sobol(scramble=True).random_base2, first count points",
               "points": pts}
        with open(os.path.join(out_dir, f"{system}_{name}.json"), "w") as f:
            json.dump(rec, f, indent=1)
        print(f"{system:14s} {name:16s} {len(pts):4d} points, seed {seed}")


if __name__ == "__main__":
    main()
