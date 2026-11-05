"""Gate on the design-verification results (run after verify/run_checks.py).

Two checks:

1. PASS CRITERIA on a results file (default: verify/results_run_checks.json). These are the
   properties the formulation claims; CI regenerates the file from scratch and applies them.
2. NUMBER BINDINGS: every number that docs/formulation_3B.md quotes from the results file is
   listed below with the exact text it appears as. The gate fails if the text is missing from
   the document, or if the committed value does not round to it. Run with --bindings on the
   committed file (CI does both).

Usage:
    python verify/check_results.py [results.json] [--bindings]
Exit status 0 only if every check passes.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def get(d, path):
    for k in path.split("/"):
        d = d[k]
    return float(d)


# (json path, test, threshold, meaning)
CRITERIA = [
    ("V1_paper1_limit/max_rel_diff_c", "lt", 1e-8, "c_w = 0 reproduces Paper 1 in c/c_in"),
    ("V1_paper1_limit/max_rel_diff_q", "lt", 1e-7, "c_w = 0 reproduces Paper 1 in q/q_max"),
    ("V1_paper1_limit/max_abs_diff_T_K", "lt", 1e-5, "c_w = 0 reproduces Paper 1 in T (K)"),
    ("V2_cycle/max_rel_rate_residual/water", "lt", 1e-12, "water-rate identity"),
    ("V2_cycle/max_rel_rate_residual/energy", "lt", 1e-12, "energy-rate identity"),
    ("V2_cycle/max_rel_rate_residual/entropy_identity", "lt", 1e-12, "dS/dt = port flux + total production"),
    ("V2_cycle/min_production_over_mean_production/ldf", "ge", -1e-12, "LDF production >= 0"),
    ("V2_cycle/min_production_over_mean_production/face", "ge", -1e-12, "dispersion + conduction production >= 0"),
    ("V2_cycle/min_production_over_mean_production/adv", "ge", -1e-12, "upwind mixing production >= 0"),
    ("V2_cycle/min_production_over_mean_production/wall", "ge", -1e-12, "wall-film production >= 0"),
    ("V3_concavity/max_eig_over_max_abs_eig_stream_entropy", "lt", 1e-4, "stream entropy concave"),
    ("V3_concavity/max_eig_over_max_abs_eig_cell_entropy", "lt", 1e-4, "cell entropy concave"),
    ("V4_coarse_graining/sigma_c_min_over_scale_fine_outlet", "gt", 0.0, "coarse entropy inequality held"),
    ("V4_coarse_graining/water_rate_rel_fine_outlet", "lt", 1e-12, "coarse water balance with the fine outlet"),
    ("V4_coarse_graining/energy_rate_rel_fine_ports", "lt", 1e-12, "coarse energy balance with the fine ports"),
    ("V4_coarse_graining/water_rate_rel_coarse_outlet", "gt", 1e-3, "the coarse outlet is the wrong port (design reason)"),
]

# (json path, exact text in docs/formulation_3B.md, value the text states, half-unit of its last digit)
BINDINGS = [
    ("V1_paper1_limit/max_rel_diff_c", "7×10⁻¹⁰ in c/c_in", 7e-10, 0.5e-10),
    ("V1_paper1_limit/max_rel_diff_q", "6×10⁻⁹ in q/q_max", 6e-9, 0.5e-9),
    ("V1_paper1_limit/max_abs_diff_T_K", "1.5×10⁻⁷ K in T", 1.5e-7, 0.05e-7),
    ("V1_paper1_limit/T_rise_K", "temperature rise 16.7 K", 16.7, 0.05),
    ("V2_cycle/max_rel_rate_residual/water", "water-rate identity 6×10⁻¹⁵", 6e-15, 0.5e-15),
    ("V2_cycle/max_rel_rate_residual/energy", "energy-rate identity 3×10⁻¹⁶", 3e-16, 0.5e-16),
    ("V2_cycle/max_rel_rate_residual/entropy_identity", "`dS/dt = Φ_S + Σσ` 3×10⁻¹⁶", 3e-16, 0.5e-16),
    ("V2_cycle/min_production_over_mean_production/adv", "worst −7×10⁻²²", -7e-22, 0.5e-22),
    ("V3_concavity/max_eig_over_max_abs_eig_stream_entropy", "ratio −2×10⁻⁵", -2e-5, 0.5e-5),
    ("V3_concavity/max_eig_over_max_abs_eig_cell_entropy", "and −3×10⁻¹⁰", -3e-10, 0.5e-10),
    ("V4_coarse_graining/sigma_c_min_over_scale_fine_outlet", "smallest margin +0.012 of scale", 0.012, 0.0005),
    ("V4_coarse_graining/wall_heat_floor_over_energy_rate", "wall-heat floor 3.3×10⁻⁵", 3.3e-5, 0.05e-5),
    ("V4_coarse_graining/max_abs_blockT_minus_mean_fineT_K", "within 0.016 K", 0.016, 0.0005),
    ("V4_coarse_graining/water_rate_rel_coarse_outlet", "off by 4.0 %", 0.040, 0.0005),
    ("V4_coarse_graining/energy_rate_rel_coarse_ports", "energy balance by 0.5 %", 0.005, 0.0005),
    ("V4_coarse_graining/snapshots", "all 602 snapshots", 602, 0.5),
]


def check_criteria(res):
    ok = True
    for path, test, thr, meaning in CRITERIA:
        v = get(res, path)
        passed = {"lt": v < thr, "gt": v > thr, "ge": v >= thr}[test]
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'}  {meaning:55s} {v: .3e} ({test} {thr:g})")
    return ok


def check_bindings(res):
    doc = open(os.path.join(ROOT, "docs", "formulation_3B.md"), encoding="utf-8").read()
    ok = True
    for path, text, stated, half in BINDINGS:
        v = get(res, path)
        present = text in doc
        rounds = abs(v - stated) <= half * (1 + 1e-9)
        passed = present and rounds
        ok &= passed
        why = "" if passed else (" (text missing from formulation)" if not present else f" (value {v:.4g} does not round to {stated:g})")
        print(f"{'PASS' if passed else 'FAIL'}  bound: {text!r}{why}")
    return ok


def main(argv):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    path = next((a for a in argv if not a.startswith("--")), os.path.join(HERE, "results_run_checks.json"))
    res = json.load(open(path))
    ok = check_criteria(res)
    if "--bindings" in argv:
        ok &= check_bindings(res)
    print("ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
