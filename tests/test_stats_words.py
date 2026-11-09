"""Plan 3, Task 7: verdict words, attribution, the coarse-solver statement, consistency, headline."""
import itertools
import math

import pytest

from p3b.stats.words import (LABELS, attribution, coarse_solver_statement, consistency, consistency_h4,
                             directional_word, h4_word, headline)

LOG_M = math.log(1.10)


@pytest.mark.parametrize("a,b,ratio,ci,word", [
    (True, True, 0.70, (0.6, 0.85), "SUPPORTED"),
    (True, True, 0.90, (0.82, 0.97), "SMALL"),
    (False, False, 0.98, (0.85, 1.10), "REFUTED"),
    (False, False, 1.30, (1.10, 1.50), "REFUTED (REVERSED)"),
    (False, True, 0.70, (0.6, 0.85), "INCONCLUSIVE"),
    (False, False, 0.80, (0.60, 1.00), "INCONCLUSIVE"),
])
def test_directional_words(a, b, ratio, ci, word):
    assert directional_word(a, b, ratio, ci) == word


@pytest.mark.parametrize("equiv,ci,demoted,word", [
    (True, (-0.05, 0.05), False, "EQUIVALENT"),
    (False, (LOG_M + 0.01, 0.4), False, "S WORSE"),
    (False, (-0.4, -LOG_M - 0.01), False, "S BETTER"),
    (False, (-0.2, 0.2), False, "INCONCLUSIVE"),
    (True, (-0.05, 0.05), True, "unresolved"),
])
def test_h4_words(equiv, ci, demoted, word):
    assert h4_word(equiv, ci, LOG_M, demoted) == word


def test_attribution():
    assert attribution({"U+F": True, "U+F+proj": True, "S-L": True}, False, False) == \
        ["the constraint", "the constraint in training", "fixed physical potentials"]
    assert attribution({"U+F": False, "U+F+proj": False, "S-L": False}, True, True) == \
        ["the physical information, not the constraint"]
    assert attribution({"U+F": False, "U+F+proj": False, "S-L": False}, True, False) == \
        ["the information suffices; the constraint's added value is unresolved"]
    assert attribution({"U+F": False, "U+F+proj": False, "S-L": False}, False, False) == \
        ["not separable from the information at this power"]


def test_coarse_solver_statement():
    assert coarse_solver_statement(True, False) == "useful against the coarse solver"
    assert coarse_solver_statement(False, True) == \
        "a coarse consistent solver makes the learned model unnecessary for this system"
    assert coarse_solver_statement(False, False) == "unresolved against the coarse solver"
    with pytest.raises(ValueError):
        coarse_solver_statement(True, True)


def test_consistency_table():
    assert consistency("SUPPORTED", "SUPPORTED") == "CONSISTENT"
    assert consistency("SUPPORTED", "SMALL") == "PARTLY CONSISTENT"
    assert consistency("SMALL", "REFUTED") == "INCONSISTENT"
    assert consistency("REFUTED", "REFUTED (REVERSED)") == "CONSISTENT"
    assert consistency("INCONCLUSIVE", "INCONCLUSIVE") == "NOT INFORMATIVE"
    assert consistency_h4("EQUIVALENT", "S BETTER") == "PARTLY CONSISTENT"
    assert consistency_h4("S WORSE", "EQUIVALENT") == "INCONSISTENT"
    assert consistency_h4("S WORSE", "S WORSE") == "CONSISTENT"
    assert consistency_h4("INCONCLUSIVE", "S WORSE") == "NOT INFORMATIVE"


V = ["SUPPORTED", "SMALL", "REFUTED", "INCONCLUSIVE"]
H4 = ["EQUIVALENT", "S BETTER", "S WORSE", "unresolved"]


def test_headline_partitions_all_256_cases_and_matches_the_prereg_labels():
    seen = set()
    for g, h1, h2, h4, h3a in itertools.product([False, True], V, V, H4, ["pass", "defect"]):
        row, text = headline(g, h1, h2, h4, h3a)
        seen.add(row)
        if g:
            assert row == "1"
        if row in ("8a", "10a"):
            assert h1 == h2 == "REFUTED" and h3a == "pass"
        if row in ("2", "3", "4", "5", "6", "7"):
            assert {"SUPPORTED", "SMALL"} & {h1, h2}
    assert seen == set(LABELS)


def test_reversed_counts_as_refuted_in_the_headline():
    assert headline(False, "REFUTED (REVERSED)", "REFUTED", "EQUIVALENT", "pass")[0] == "8a"
