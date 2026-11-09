"""Check the pre-registered headline decision list (PREREG_P3B_draft.md, Section 8).

Enumerates every combination of the gate outcome (pass, fail), the H1 and H2 verdicts
(SUPPORTED, SMALL, REFUTED, INCONCLUSIVE; NOT TESTABLE counts as INCONCLUSIVE), the H4 outcome
(EQUIVALENT, S BETTER, S WORSE, unresolved = INCONCLUSIVE or demoted) and H3a (pass, defect).
Asserts:
  1. A, A_s and R are mutually exclusive;
  2. a failed gate gives row 1 and nothing else;
  3. every other combination is matched by exactly one of rows 2 to 13, or falls to row 14;
  4. "admissibility" (rows 8a, 10a) fires only when H1 and H2 are both REFUTED and H3a = pass,
     and 8a only with H4 EQUIVALENT;
  5. no row that credits structure (2 to 7) fires unless H1 or H2 is SUPPORTED or SMALL;
  6. the row labels of the PREREG table are exactly the labels checked here.
Prints the row counts. Exit status 0 only if every assertion holds.
"""
import itertools
import os
import re
import sys
from collections import Counter

V = ["SUPPORTED", "SMALL", "REFUTED", "INCONCLUSIVE"]
H4 = ["EQUIVALENT", "S BETTER", "S WORSE", "unresolved"]
H3A = ["pass", "defect"]
GATE = ["pass", "fail"]
LABELS = ["1", "2", "3", "4", "5", "6", "7", "8a", "8b", "9", "10a", "10b", "11", "12", "13", "14"]
PREREG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "PREREG_P3B_draft.md")


def rows_matching(gate, h1, h2, h4, h3a):
    if gate == "fail":
        return ["1"]
    A = "SUPPORTED" in (h1, h2)
    As = (not A) and ("SMALL" in (h1, h2))
    R = h1 == "REFUTED" and h2 == "REFUTED"
    assert A + As + R <= 1, (h1, h2, h4)
    none = not (A or As or R)
    conds = {
        "2": A and h4 in ("EQUIVALENT", "S BETTER"),
        "3": A and h4 == "S WORSE",
        "4": A and h4 == "unresolved",
        "5": As and h4 in ("EQUIVALENT", "S BETTER"),
        "6": As and h4 == "S WORSE",
        "7": As and h4 == "unresolved",
        "8a": R and h4 == "EQUIVALENT" and h3a == "pass",
        "8b": R and h4 == "EQUIVALENT" and h3a == "defect",
        "9": R and h4 == "S BETTER",
        "10a": R and h4 == "S WORSE" and h3a == "pass",
        "10b": R and h4 == "S WORSE" and h3a == "defect",
        "11": R and h4 == "unresolved",
        "12": none and h4 == "S BETTER",
        "13": none and h4 == "S WORSE",
    }
    return [r for r, c in conds.items() if c]


def prereg_labels(path=PREREG):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    sec = text.split("## 8.", 1)[1].split("\n## ", 1)[0]
    return [m.group(1) for m in re.finditer(r"^\| (\d+[ab]?) \|", sec, flags=re.M)]


def main():
    counts = Counter()
    for gate, h1, h2, h4, h3a in itertools.product(GATE, V, V, H4, H3A):
        m = rows_matching(gate, h1, h2, h4, h3a)
        assert len(m) <= 1, ("overlap", gate, h1, h2, h4, h3a, m)
        row = m[0] if m else "14"
        counts[row] += 1
        if gate == "fail":
            assert row == "1"
        if row in ("8a", "10a"):
            assert h1 == h2 == "REFUTED" and h3a == "pass"
        if row == "8a":
            assert h4 == "EQUIVALENT"
        if row in ("2", "3", "4", "5", "6", "7"):
            assert {"SUPPORTED", "SMALL"} & {h1, h2}
    total = sum(counts.values())
    assert total == 2 * 64 * 2
    assert set(counts) <= set(LABELS)
    labels = prereg_labels()
    assert labels == LABELS, ("PREREG table labels differ", labels)
    print("combinations:", total)
    print("row counts:", {k: counts[k] for k in LABELS if counts[k]})
    print("PASS: ordered partition; gate failures give row 1; admissibility only when both REFUTED and H3a = pass; "
          "credit rows need SUPPORTED or SMALL; PREREG labels match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
