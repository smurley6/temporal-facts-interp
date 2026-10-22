"""Temporal relations drawn from Wikidata.

Each relation is a Wikidata property whose statements carry start (P580) and end (P582)
qualifiers, so the full change history of a fact is machine-readable. The prompt template is
written for a base (non-chat) model: a declarative sentence the model completes.

`subject_filter` is a SPARQL fragment binding ?s; `office_prop` (optional) names the property
whose value label replaces `{office}` in the template (used for head-of-government style
relations where the title differs per subject: "Prime Minister of Canada", "Mayor of Paris").
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Relation:
    key: str
    pid: str
    description: str
    template: str
    min_sitelinks: int
    subject_filter: str = ""
    office_prop: str | None = None
    answer_kind: str = "person"  # person | place | organization


RELATIONS: dict[str, Relation] = {
    "ceo": Relation(
        key="ceo",
        pid="P169",
        description="chief executive officer",
        template="The chief executive officer of {subject} is",
        min_sitelinks=20,
    ),
    "chair": Relation(
        key="chair",
        pid="P488",
        description="chairperson",
        template="The chairperson of {subject} is",
        min_sitelinks=25,
    ),
    "head_coach": Relation(
        key="head_coach",
        pid="P286",
        description="head coach",
        template="The head coach of {subject} is",
        min_sitelinks=20,
    ),
    "head_of_government": Relation(
        key="head_of_government",
        pid="P6",
        description="head of government (title from P1313)",
        template="The {office} is",
        min_sitelinks=40,
        office_prop="P1313",
    ),
    "head_of_state": Relation(
        key="head_of_state",
        pid="P35",
        description="head of state (title from P1906)",
        template="The {office} is",
        min_sitelinks=60,
        office_prop="P1906",
    ),
    "director": Relation(
        key="director",
        pid="P1037",
        description="director / manager of an institution",
        template="The director of {subject} is",
        min_sitelinks=30,
    ),
    "headquarters": Relation(
        key="headquarters",
        pid="P159",
        description="headquarters location",
        template="The headquarters of {subject} is located in",
        min_sitelinks=20,
        subject_filter="?s wdt:P169 [] .",
        answer_kind="place",
    ),
}
