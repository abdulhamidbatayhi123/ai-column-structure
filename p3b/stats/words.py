"""Verdict words, attribution, the coarse-solver statement, consistency words and the headline
(PREREG Sections 6, 7 and 8; Plan 3, Task 7). `verify/check_decision_list.py` checks ROWS."""
from __future__ import annotations

REFUTED = "REFUTED"


def _base(word: str) -> str:
    return REFUTED if word.startswith(REFUTED) else word


def directional_word(a_holds: bool, b_holds: bool, ratio: float, ci_ratio) -> str:
    """H1, H2, H5, H6, H3b: the first word that applies (PREREG 6). ci_ratio is the (1 - 2 alpha_h)
    interval of the ratio exp(HL)."""
    lo, _ = ci_ratio
    if a_holds and b_holds and ratio <= 0.8:
        return "SUPPORTED"
    if a_holds and b_holds and 0.8 < ratio < 1.0:
        return "SMALL"
    if lo > 0.8:
        return "REFUTED (REVERSED)" if lo > 1.0 else REFUTED
    return "INCONCLUSIVE"


def h4_word(tost_equivalent: bool, ci90_log, log_m: float, demoted: bool = False) -> str:
    if demoted:
        return "unresolved"
    lo, hi = ci90_log
    if tost_equivalent:
        return "EQUIVALENT"
    if lo > log_m:
        return "S WORSE"
    if hi < -log_m:
        return "S BETTER"
    return "INCONCLUSIVE"


def attribution(s_beats: dict, uf_beats_u: bool, s_uf_equivalent: bool) -> list:
    """Labels for credited rows; s_beats maps "U+F", "U+F+proj", "S-L" to "S beats X"."""
    out = []
    if s_beats.get("U+F"):
        out.append("the constraint")
    if s_beats.get("U+F+proj"):
        out.append("the constraint in training")
    if s_beats.get("S-L"):
        out.append("fixed physical potentials")
    if not s_beats.get("U+F"):
        if uf_beats_u and s_uf_equivalent:
            out.append("the physical information, not the constraint")
        elif uf_beats_u:
            out.append("the information suffices; the constraint's added value is unresolved")
        else:
            out.append("not separable from the information at this power")
    return out


def coarse_solver_statement(superior_to_all: bool, noninferior_any: bool) -> str:
    if superior_to_all and noninferior_any:
        raise ValueError("S cannot be superior to every coarse grid while one is non-inferior to S")
    if superior_to_all:
        return "useful against the coarse solver"
    if noninferior_any:
        return "a coarse consistent solver makes the learned model unnecessary for this system"
    return "unresolved against the coarse solver"


_TABLE = {
    "SUPPORTED": {"SUPPORTED": "CONSISTENT", "SMALL": "PARTLY CONSISTENT", REFUTED: "INCONSISTENT"},
    "SMALL": {"SUPPORTED": "PARTLY CONSISTENT", "SMALL": "CONSISTENT", REFUTED: "INCONSISTENT"},
    REFUTED: {"SUPPORTED": "INCONSISTENT", "SMALL": "INCONSISTENT", REFUTED: "CONSISTENT"},
}


def consistency(p: str, o: str) -> str:
    return _TABLE.get(_base(p), {}).get(_base(o), "NOT INFORMATIVE")


def consistency_h4(p: str, o: str) -> str:
    if "INCONCLUSIVE" in (p, o) or "unresolved" in (p, o):
        return "NOT INFORMATIVE"
    if p == o:
        return "CONSISTENT"
    if {p, o} == {"EQUIVALENT", "S BETTER"}:
        return "PARTLY CONSISTENT"
    return "INCONSISTENT"


ROWS = [
    ("2", lambda A, As, R, n, h4, h3a: A and h4 in ("EQUIVALENT", "S BETTER"), "Structure pays in [H]."),
    ("3", lambda A, As, R, n, h4, h3a: A and h4 == "S WORSE",
     "Structure pays in [H], at a price in short-horizon accuracy."),
    ("4", lambda A, As, R, n, h4, h3a: A and h4 == "unresolved",
     "Structure pays in [H]; its accuracy cost is unresolved at this power."),
    ("5", lambda A, As, R, n, h4, h3a: As and h4 in ("EQUIVALENT", "S BETTER"),
     "Structure helps in [H], but less than the declared minimum."),
    ("6", lambda A, As, R, n, h4, h3a: As and h4 == "S WORSE",
     "Structure helps in [H] less than the declared minimum, at a price in accuracy."),
    ("7", lambda A, As, R, n, h4, h3a: As and h4 == "unresolved",
     "Structure helps in [H] less than the declared minimum; its accuracy cost is unresolved."),
    ("8a", lambda A, As, R, n, h4, h3a: R and h4 == "EQUIVALENT" and h3a == "pass",
     "Structure buys admissibility only."),
    ("8b", lambda A, As, R, n, h4, h3a: R and h4 == "EQUIVALENT" and h3a == "defect",
     "Structure buys neither stability nor data efficiency, and its admissibility guarantee had "
     "projection defects."),
    ("9", lambda A, As, R, n, h4, h3a: R and h4 == "S BETTER",
     "Structure improves short-horizon accuracy but buys neither stability nor data efficiency."),
    ("10a", lambda A, As, R, n, h4, h3a: R and h4 == "S WORSE" and h3a == "pass",
     "Structure costs accuracy and buys only admissibility."),
    ("10b", lambda A, As, R, n, h4, h3a: R and h4 == "S WORSE" and h3a == "defect",
     "Structure costs accuracy and buys neither stability nor data efficiency; its admissibility "
     "guarantee had projection defects."),
    ("11", lambda A, As, R, n, h4, h3a: R and h4 == "unresolved",
     "Structure buys neither stability nor data efficiency; its accuracy cost is unresolved."),
    ("12", lambda A, As, R, n, h4, h3a: n and h4 == "S BETTER",
     "Structure improves short-horizon accuracy; stability and data efficiency are unresolved at "
     "this power."),
    ("13", lambda A, As, R, n, h4, h3a: n and h4 == "S WORSE",
     "Structure costs accuracy; its benefits are unresolved at this power."),
]
LABELS = ["1"] + [r[0] for r in ROWS] + ["14"]


def headline(gate_failed: bool, h1: str, h2: str, h4: str, h3a: str):
    """(row label, headline text) by the ordered decision list; H4 "INCONCLUSIVE" counts as
    "unresolved"."""
    if gate_failed:
        return "1", "the gate outcome only"
    h1, h2 = _base(h1), _base(h2)
    h4 = "unresolved" if h4 in ("INCONCLUSIVE", "unresolved") else h4
    A = "SUPPORTED" in (h1, h2)
    As = (not A) and "SMALL" in (h1, h2)
    R = h1 == REFUTED and h2 == REFUTED
    none = not (A or As or R)
    hits = [(lab, text) for lab, cond, text in ROWS if cond(A, As, R, none, h4, h3a)]
    if len(hits) > 1:
        raise AssertionError(f"rows overlap: {[h[0] for h in hits]}")
    return hits[0] if hits else ("14", "No measurable benefit at this power")
