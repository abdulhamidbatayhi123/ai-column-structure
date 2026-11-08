"""Apply the reference gates to every trajectory npz in a directory.

    python scripts/check_reference_gate.py data/reference/A_primary/train

Prints one PASS/FAIL line per gate and file (closure, entropy margin, floor share), an INFO line
with the cyclic-steady-state cycle, and exits non-zero if any gate fails.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from p3b import gates  # noqa: E402


def main(directory: str) -> int:
    files = sorted(glob.glob(os.path.join(directory, "*.npz")))
    if not files:
        print(f"FAIL no npz files in {directory}")
        return 1
    failed = 0
    for f in files:
        name = os.path.basename(f)
        meta = gates.load_meta(f)
        checks = [
            ("closure", gates.stored_closure(f), lambda x: x <= gates.CLOSURE_TOL, f"<= {gates.CLOSURE_TOL:g}"),
            ("entropy_margin", gates.coarse_entropy_margin(f, gates.model_from_meta(meta)), lambda x: x > 0, "> 0"),
            ("floor_share", gates.floor_share(f), lambda x: x < gates.FLOOR_SHARE_TOL, f"< {gates.FLOOR_SHARE_TOL:g}"),
        ]
        for gate, value, ok, rule in checks:
            failed += not ok(value)
            print(f"{'PASS' if ok(value) else 'FAIL'} {gate:15s} {name} {value:.3e} ({rule})")
        print(f"INFO css_cycle       {name} {gates.css_cycle(f)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
