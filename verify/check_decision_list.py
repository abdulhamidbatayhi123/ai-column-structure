"""Check the pre-registered headline decision list (PREREG_P3B_draft.md, Section 8).

Enumerates every combination of the H1 and H2 verdicts (SUPPORTED, SMALL, REFUTED,
INCONCLUSIVE; NOT TESTABLE counts as INCONCLUSIVE) with the H4 outcome (EQUIVALENT,
S BETTER, S WORSE, unresolved = INCONCLUSIVE or demoted). Asserts:
  1. A, A_s and R are mutually exclusive;
  2. every combination is matched by exactly one of rows 2 to 13, or falls to row 14;
  3. "admissibility only" (row 8) fires only when H1 and H2 are both REFUTED and H4 is
     EQUIVALENT;
  4. no row that credits structure (2 to 7) fires unless H1 or H2 is SUPPORTED or SMALL.
Prints the row counts. Exit status 0 only if every assertion holds.
"""
import itertools
import sys
from collections import Counter

V = ["SUPPORTED", "SMALL", "REFUTED", "INCONCLUSIVE"]
H4 = ["EQUIVALENT", "S BETTER", "S WORSE", "unresolved"]


def rows_matching(h1, h2, h4):
    A = "SUPPORTED" in (h1, h2)
    As = (not A) and ("SMALL" in (h1, h2))
    R = h1 == "REFUTED" and h2 == "REFUTED"
    assert A + As + R <= 1, (h1, h2, h4)
    none = not (A or As or R)
    conds = {
        2: A and h4 in ("EQUIVALENT", "S BETTER"),
        3: A and h4 == "S WORSE",
        4: A and h4 == "unresolved",
        5: As and h4 in ("EQUIVALENT", "S BETTER"),
        6: As and h4 == "S WORSE",
        7: As and h4 == "unresolved",
        8: R and h4 == "EQUIVALENT",
        9: R and h4 == "S BETTER",
        10: R and h4 == "S WORSE",
        11: R and h4 == "unresolved",
        12: none and h4 == "S BETTER",
        13: none and h4 == "S WORSE",
    }
    return [r for r, c in conds.items() if c]


def main():
    counts = Counter()
    for h1, h2, h4 in itertools.product(V, V, H4):
        m = rows_matching(h1, h2, h4)
        assert len(m) <= 1, ("overlap", h1, h2, h4, m)
        row = m[0] if m else 14
        counts[row] += 1
        if row == 8:
            assert h1 == h2 == "REFUTED" and h4 == "EQUIVALENT"
        if row in range(2, 8):
            assert {"SUPPORTED", "SMALL"} & {h1, h2}
    total = sum(counts.values())
    assert total == 64
    print("combinations:", total)
    print("row counts:", dict(sorted(counts.items())))
    print("PASS: ordered partition; row 8 only when both REFUTED and EQUIVALENT; credit rows need SUPPORTED or SMALL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
