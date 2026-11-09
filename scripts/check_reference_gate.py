"""Apply the reference gates to every trajectory npz in a directory.

    python scripts/check_reference_gate.py data/reference/A_primary/train

Prints one PASS/FAIL line per gate and file (production-ready, integrated and stored closure, rate
identities, fine production, the entropy gate, floor share), an INFO line
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
        ready, why = gates.production_ready(f)
        eg = gates.entropy_gate(f, gates.model_from_meta(meta))
        checks = [
            ("production", 0.0 if ready else 1.0, lambda x: x == 0.0, why or "not a smoke output"),
            ("closure_integ", float(meta["closure_max"]), lambda x: x <= gates.CLOSURE_TOL, f"<= {gates.CLOSURE_TOL:g}"),
            ("closure_stored", gates.stored_closure(f), lambda x: x <= gates.CLOSURE_TOL, f"<= {gates.CLOSURE_TOL:g}"),
            ("rate_identity", gates.rate_identity_max(f), lambda x: x <= 1e-12, "<= 1e-12"),
            ("fine_production", gates.fine_production_min(f), lambda x: x >= -1e-12, ">= -1e-12"),
            ("entropy_gate", eg["min_rate"], lambda x: eg["passed"], f">= -tau_ref = {-eg['tau_ref']:.3e}, {eg['tau_source']}"),
            ("floor_share", gates.floor_share(f), lambda x: x < gates.FLOOR_SHARE_TOL, f"< {gates.FLOOR_SHARE_TOL:g}"),
        ]
        for gate, value, ok, rule in checks:
            good = bool(value == value and ok(value))      # a NaN fails
            failed += not good
            print(f"{'PASS' if good else 'FAIL'} {gate:15s} {name} {value:.3e} ({rule})")
        print(f"INFO css_cycle       {name} {gates.css_cycle(f)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
