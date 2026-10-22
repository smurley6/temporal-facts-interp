"""Pure labeling logic: a fact's Wikidata statement history -> shelf-life class.

Classes (relative to the model's training cutoff C and the snapshot date T):

  stable       current value began at least STABLE_YEARS before C and still holds at T.
  superseded   a different value held at C (start <= C < end) and the current value
               began after C. The model saw the old value in training; the fact expired.
  recent       current value began within RECENT_MONTHS before C and still holds at T.
               Present in training data, but thinly.

Everything else is excluded with an explicit reason so the exclusion is auditable.

Dates near the cutoff with only year precision are ambiguous and are excluded rather than
guessed. Wikidata time precision: 9 = year, 10 = month, 11 = day.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

STABLE_YEARS = 5
RECENT_MONTHS = 18
MIN_AGE_DAYS_AFTER_CHANGE = 60  # a change must be at least this old at snapshot time


@dataclass(frozen=True)
class Statement:
    statement_id: str
    value_qid: str
    value_label: str
    rank: str  # "preferred" | "normal"
    start: date | None
    start_precision: int | None
    end: date | None
    end_precision: int | None


@dataclass
class LabelResult:
    label: str  # stable | superseded | recent | excluded
    reason: str = ""
    current: Statement | None = None
    previous: Statement | None = None  # value that held at the cutoff (superseded only)
    notes: list[str] = field(default_factory=list)


def _is_current(st: Statement, today: date) -> bool:
    return st.end is None or st.end > today


def _ambiguous_near(d: date, precision: int | None, cutoff: date, window_days: int = 366) -> bool:
    """Year-precision (or missing-precision) dates within a year of the cutoff are unusable."""
    if precision is not None and precision >= 10:
        return False
    return abs((d - cutoff).days) <= window_days


def label_fact(statements: list[Statement], *, cutoff: date, today: date) -> LabelResult:
    sts = [s for s in statements if s.rank != "deprecated"]
    if not sts:
        return LabelResult("excluded", "no_statements")

    current = [s for s in sts if _is_current(s, today)]
    if not current:
        return LabelResult("excluded", "no_current_value")
    if len(current) > 1:
        preferred = [s for s in current if s.rank == "preferred"]
        if len(preferred) == 1:
            current = preferred
        else:
            distinct = {s.value_qid for s in current}
            if len(distinct) > 1:
                return LabelResult("excluded", "multiple_current_values")
            # duplicate statements for the same value: keep the earliest-started one
            current = [min(current, key=lambda s: s.start or date.min)]
    cur = current[0]

    if cur.start is None:
        return LabelResult("excluded", "current_value_undated", current=cur)
    if cur.start > today:
        return LabelResult("excluded", "current_value_starts_in_future", current=cur)

    others = [s for s in sts if s is not cur and s.value_qid != cur.value_qid]

    # superseded: current started after the cutoff and some other value held at the cutoff
    if cur.start > cutoff:
        if _ambiguous_near(cur.start, cur.start_precision, cutoff):
            return LabelResult("excluded", "change_date_imprecise_near_cutoff", current=cur)
        if (today - cur.start).days < MIN_AGE_DAYS_AFTER_CHANGE:
            return LabelResult("excluded", "change_too_fresh", current=cur)
        held_at_cutoff = [
            s
            for s in others
            if s.start is not None and s.start <= cutoff and (s.end is None or s.end > cutoff)
        ]
        if not held_at_cutoff:
            return LabelResult("excluded", "no_predecessor_at_cutoff", current=cur)
        if len({s.value_qid for s in held_at_cutoff}) > 1:
            return LabelResult("excluded", "multiple_predecessors_at_cutoff", current=cur)
        prev = held_at_cutoff[0]
        if prev.end is not None and _ambiguous_near(prev.end, prev.end_precision, cutoff):
            return LabelResult("excluded", "predecessor_end_imprecise_near_cutoff", current=cur)
        return LabelResult("superseded", current=cur, previous=prev)

    # value began on or before the cutoff and still holds
    age_at_cutoff = cutoff - cur.start
    if age_at_cutoff >= timedelta(days=365 * STABLE_YEARS):
        return LabelResult("stable", current=cur)
    if age_at_cutoff <= timedelta(days=30 * RECENT_MONTHS):
        if _ambiguous_near(cur.start, cur.start_precision, cutoff):
            return LabelResult("excluded", "recent_start_imprecise_near_cutoff", current=cur)
        return LabelResult("recent", current=cur)
    return LabelResult("excluded", "intermediate_age", current=cur)
