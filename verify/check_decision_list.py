"""Check the pre-registered headline decision list (PREREG_P3B_draft.md, Section 8) as the code
implements it: p3b.stats.words.headline, the one decision list the analysis uses.

Enumerates every combination of the gate outcome (pass, fail), the H1 and H2 verdicts
(SUPPORTED, SMALL, REFUTED, REFUTED (REVERSED), INCONCLUSIVE, NOT TESTABLE), the H4 outcome
(EQUIVALENT, S BETTER, S WORSE, INCONCLUSIVE, unresolved) and H3a (pass, defect). Asserts:
  1. a failed gate gives row 1 and nothing else;
  2. every other combination is matched by exactly one of rows 2 to 13, or falls to row 14
     (headline raises on overlapping rows);
  3. "admissibility" (rows 8a, 10a) fires only when H1 and H2 are both REFUTED and H3a = pass,
     and 8a only with H4 EQUIVALENT;
  4. no row that credits structure (2 to 7) fires unless H1 or H2 is SUPPORTED or SMALL;
  5. the row labels of the PREREG table are exactly the code's labels, in order.
Prints the row counts. Exit status 0 only if every assertion holds.
"""
import itertools
import os
import re
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from p3b.stats.words import LABELS, headline  # noqa: E402

V = ["SUPPORTED", "SMALL", "REFUTED", "REFUTED (REVERSED)", "INCONCLUSIVE", "NOT TESTABLE"]
H4 = ["EQUIVALENT", "S BETTER", "S WORSE", "INCONCLUSIVE", "unresolved"]
H3A = ["pass", "defect"]
PREREG = os.path.join(ROOT, "PREREG_P3B_draft.md")


def prereg_labels(path=PREREG):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    sec = text.split("## 8.", 1)[1].split("\n## ", 1)[0]
    return [m.group(1) for m in re.finditer(r"^\| (\d+[ab]?) \|", sec, flags=re.M)]


def main():
    counts = Counter()
    for gate, h1, h2, h4, h3a in itertools.product([False, True], V, V, H4, H3A):
        row, _ = headline(gate, h1, h2, h4, h3a)
        counts[row] += 1
        base = lambda w: "REFUTED" if w.startswith("REFUTED") else w  # noqa: E731
        if gate:
            assert row == "1"
        if row in ("8a", "10a"):
            assert base(h1) == base(h2) == "REFUTED" and h3a == "pass"
        if row == "8a":
            assert h4 == "EQUIVALENT"
        if row in ("2", "3", "4", "5", "6", "7"):
            assert {"SUPPORTED", "SMALL"} & {h1, h2}
    total = sum(counts.values())
    assert total == 2 * len(V) ** 2 * len(H4) * len(H3A)
    labels = prereg_labels()
    assert labels == LABELS, ("PREREG table labels differ from the code", labels, LABELS)
    print("combinations:", total)
    print("row counts:", {k: counts[k] for k in LABELS if counts[k]})
    print("PASS: the code's decision list (p3b.stats.words.headline) partitions every case; gate failures give "
          "row 1; admissibility only when both REFUTED and H3a = pass; credit rows need SUPPORTED or SMALL; "
          "PREREG labels match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
