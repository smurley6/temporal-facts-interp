from datetime import date

from tfi.labeling import Statement, label_fact

C = date(2023, 12, 31)
T = date(2026, 10, 21)


def st(sid, qid, start, end=None, rank="normal", sp=11, ep=11):
    return Statement(sid, qid, qid, rank, start, sp if start else None, end, ep if end else None)


def test_stable():
    r = label_fact([st("a", "Q1", date(2011, 8, 24))], cutoff=C, today=T)
    assert r.label == "stable"


def test_superseded_with_predecessor():
    r = label_fact(
        [st("a", "Q1", date(2011, 8, 24), date(2025, 8, 31)), st("b", "Q2", date(2025, 9, 1))],
        cutoff=C,
        today=T,
    )
    assert r.label == "superseded"
    assert r.current.value_qid == "Q2" and r.previous.value_qid == "Q1"


def test_superseded_requires_predecessor_at_cutoff():
    r = label_fact([st("b", "Q2", date(2025, 1, 1))], cutoff=C, today=T)
    assert r.label == "excluded" and r.reason == "no_predecessor_at_cutoff"


def test_change_too_fresh_excluded():
    r = label_fact(
        [st("a", "Q1", date(2010, 1, 1), date(2026, 10, 1)), st("b", "Q2", date(2026, 10, 2))],
        cutoff=C,
        today=T,
    )
    assert r.reason == "change_too_fresh"


def test_year_precision_near_cutoff_excluded():
    r = label_fact(
        [
            st("a", "Q1", date(2010, 1, 1), date(2024, 1, 1), ep=9),
            st("b", "Q2", date(2024, 1, 1), sp=9),
        ],
        cutoff=C,
        today=T,
    )
    assert r.label == "excluded" and "imprecise" in r.reason


def test_recent():
    r = label_fact([st("a", "Q1", date(2023, 3, 1))], cutoff=C, today=T)
    assert r.label == "recent"


def test_intermediate_excluded():
    r = label_fact([st("a", "Q1", date(2021, 1, 1))], cutoff=C, today=T)
    assert r.reason == "intermediate_age"


def test_multiple_current_distinct_excluded():
    r = label_fact(
        [st("a", "Q1", date(2010, 1, 1)), st("b", "Q2", date(2012, 1, 1))], cutoff=C, today=T
    )
    assert r.reason == "multiple_current_values"


def test_preferred_rank_breaks_tie():
    r = label_fact(
        [st("a", "Q1", date(2010, 1, 1)), st("b", "Q2", date(2012, 1, 1), rank="preferred")],
        cutoff=C,
        today=T,
    )
    assert r.label == "stable" and r.current.value_qid == "Q2"


def test_undated_current_excluded():
    r = label_fact([st("a", "Q1", None)], cutoff=C, today=T)
    assert r.reason == "current_value_undated"
