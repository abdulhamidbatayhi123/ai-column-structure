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


DIRECTIONAL = ("SUPPORTED", "SMALL", REFUTED, "REFUTED (REVERSED)", "INCONCLUSIVE", "NOT TESTABLE")
H4_WORDS = ("EQUIVALENT", "S BETTER", "S WORSE", "INCONCLUSIVE", "unresolved")


def headline(gate_failed: bool, h1: str, h2: str, h4: str, h3a: str):
    """(row label, headline text) by the ordered decision list; H4 "INCONCLUSIVE" counts as
    "unresolved"; NOT TESTABLE counts as INCONCLUSIVE. Unknown inputs raise ValueError."""
    for name, v, allowed in (("H1", h1, DIRECTIONAL), ("H2", h2, DIRECTIONAL), ("H4", h4, H4_WORDS),
                             ("H3a", h3a, ("pass", "defect"))):
        if v not in allowed:
            raise ValueError(f"{name} = {v!r} is not one of {allowed}")
    h1 = "INCONCLUSIVE" if h1 == "NOT TESTABLE" else h1
    h2 = "INCONCLUSIVE" if h2 == "NOT TESTABLE" else h2
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


NAMES = {"H1": "long-horizon stability", "H2": "data efficiency"}
SHORT = {"H1": "stability", "H2": "data efficiency"}


def compose_headline(row: str, carrying: list, h1: str, h2: str, attribution: list, consistency: dict,
                     coarse: str, h4: str, m: float) -> str:
    """The printed headline (PREREG Section 8): the row's text with [H] filled; "+ no [stability |
    data efficiency]" and "(REVERSED)" for the other hypothesis; an EQUIVALENT H4's margin; then the
    qualifiers in order: attribution, consistency words (a SUPPORTED hypothesis INCONSISTENT on the
    second isotherm or on system B is demoted: "pays" -> "helps in one setting", naming where it
    holds), and the coarse-solver statement. consistency: {H: (word second isotherm, word system B)}."""
    text = dict((lab, t) for lab, _, t in ROWS)[row]
    words = {"H1": h1, "H2": h2}
    demoted = [h for h in carrying if words[h] == "SUPPORTED" and "INCONSISTENT" in consistency.get(h, ())]
    if demoted and "pays" in text:
        text = text.replace("pays", "helps in one setting")
        where = []
        for h in demoted:
            r, b = consistency[h]
            places = ["system A (MOF-303)"] + (["the second isotherm"] if r != "INCONSISTENT" else []) \
                + (["system B"] if b != "INCONSISTENT" else [])
            where.append(f"{SHORT[h]} holds in {', '.join(places)}")
        text = text.rstrip(".") + " (" + "; ".join(where) + ")."
    text = text.replace("[H]", " and ".join(NAMES[h] for h in carrying))
    for h in carrying:
        other = "H2" if h == "H1" else "H1"
        if other not in carrying and words[other].startswith(REFUTED):
            text = text.rstrip(".") + f" + no {SHORT[other]}" + (" (REVERSED)" if "REVERSED" in words[other] else "") + "."
    if h4 == "EQUIVALENT":
        text = text.rstrip(".") + f" (accuracy equivalent within ±{(m - 1) * 100:.0f} %)."
    quals = list(attribution)
    for h in carrying:
        if h in consistency:
            quals.append(f"{h}: second isotherm {consistency[h][0]}, system B {consistency[h][1]}")
    quals.append(coarse)
    return text + " " + "; ".join(quals) + "."


def h3b_test(drift, error, n_perm: int = 10_000, seed: int = 0):
    """H3b: Spearman's rho between balance drift and error in U over test trajectories, with a
    one-sided (positive association) permutation p-value. Described as an association."""
    import numpy as np
    from scipy import stats as st
    drift, error = np.asarray(drift, float), np.asarray(error, float)
    rho = float(st.spearmanr(drift, error).statistic)
    rng = np.random.default_rng(seed)
    perm = np.array([st.spearmanr(drift, rng.permutation(error)).statistic for _ in range(n_perm)])
    return rho, float((1 + np.sum(perm >= rho)) / (n_perm + 1))


def h6_ratio(regret_s, regret_u, floor: float = 0.01):
    """H6: the floored regret ratio (r_S + 0.01) / (r_U + 0.01) per scenario (PREREG Section 6)."""
    import numpy as np
    return (np.asarray(regret_s, float) + floor) / (np.asarray(regret_u, float) + floor)


def variance_decomposition(values) -> dict:
    """Trajectory x seed variance components of the log error ratios (two-way crossed layout without
    replication, method of moments, negative estimates set to 0), reported beside every test."""
    import numpy as np
    v = np.asarray(values, float)
    n_t, n_s = v.shape
    gm = v.mean()
    ms_t = n_s * np.sum((v.mean(1) - gm) ** 2) / (n_t - 1)
    ms_s = n_t * np.sum((v.mean(0) - gm) ** 2) / (n_s - 1)
    resid = v - v.mean(1, keepdims=True) - v.mean(0, keepdims=True) + gm
    ms_e = np.sum(resid ** 2) / ((n_t - 1) * (n_s - 1))
    return {"trajectory": float(max(0.0, (ms_t - ms_e) / n_s)), "seed": float(max(0.0, (ms_s - ms_e) / n_t)),
            "residual": float(ms_e)}
