"""Wikidata SPARQL client with frozen raw snapshots.

Every query result is written verbatim to data/raw/<name>.<timestamp>.json before any
processing, so the processed benchmark can always be regenerated from an immutable input and
every label traces back to specific Wikidata statement IDs.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "temporal-facts-interp/0.1 (https://github.com/smurley6; research use)"

STATEMENT_QUERY = """
SELECT ?s ?sLabel ?sitelinks ?st ?v ?vLabel ?rank ?start ?startPrec ?end ?endPrec {office_select}
WHERE {{
  ?s p:{pid} ?st .
  ?s wikibase:sitelinks ?sitelinks .
  FILTER(?sitelinks >= {min_sitelinks})
  ?enwiki schema:about ?s ; schema:isPartOf <https://en.wikipedia.org/> .
  {subject_filter}
  ?st ps:{pid} ?v .
  ?st wikibase:rank ?rank .
  FILTER(?rank != wikibase:DeprecatedRank)
  OPTIONAL {{ ?st pqv:P580 ?sv . ?sv wikibase:timeValue ?start ; wikibase:timePrecision ?startPrec . }}
  OPTIONAL {{ ?st pqv:P582 ?ev . ?ev wikibase:timeValue ?end ; wikibase:timePrecision ?endPrec . }}
  {office_clause}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
"""

VALUE_META_QUERY = """
SELECT ?v ?sitelinks ?enwiki WHERE {{
  VALUES ?v {{ {values} }}
  ?v wikibase:sitelinks ?sitelinks .
  OPTIONAL {{ ?enwiki schema:about ?v ; schema:isPartOf <https://en.wikipedia.org/> . }}
}}
"""

REF_COUNT_QUERY = """
SELECT ?st (COUNT(DISTINCT ?ref) AS ?refs) WHERE {{
  VALUES ?st {{ {values} }}
  OPTIONAL {{ ?st prov:wasDerivedFrom ?ref . }}
}} GROUP BY ?st
"""

ALIAS_QUERY = """
SELECT ?v ?alt WHERE {{
  VALUES ?v {{ {values} }}
  ?v skos:altLabel ?alt . FILTER(LANG(?alt) = "en")
}}
"""


def run_sparql(query: str, *, retries: int = 4, timeout: int = 120) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(
                ENDPOINT,
                params={"query": query, "format": "json"},
                headers={"Accept": "application/sparql-results+json", "User-Agent": USER_AGENT},
                timeout=timeout,
            )
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", "10"))
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:  # noqa: PERF203
            last = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"SPARQL query failed after {retries} attempts: {last}")


def freeze(raw_dir: Path, name: str, query: str, payload: dict[str, Any]) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = raw_dir / f"{name}.{stamp}.json"
    path.write_text(
        json.dumps(
            {"retrieved_at": stamp, "endpoint": ENDPOINT, "query": query, "result": payload},
            ensure_ascii=False,
        )
    )
    return path


def fetch_statements(
    *,
    pid: str,
    min_sitelinks: int,
    subject_filter: str = "",
    office_prop: str | None = None,
) -> tuple[str, dict[str, Any]]:
    office_select = "?office ?officeLabel" if office_prop else ""
    office_clause = f"?s wdt:{office_prop} ?office ." if office_prop else ""
    q = STATEMENT_QUERY.format(
        pid=pid,
        min_sitelinks=min_sitelinks,
        subject_filter=subject_filter,
        office_select=office_select,
        office_clause=office_clause,
    )
    return q, run_sparql(q)


def fetch_aliases(qids: list[str], batch: int = 300) -> tuple[list[str], dict[str, list[str]]]:
    out: dict[str, list[str]] = {}
    queries: list[str] = []
    for i in range(0, len(qids), batch):
        chunk = qids[i : i + batch]
        q = ALIAS_QUERY.format(values=" ".join(f"wd:{x}" for x in chunk))
        queries.append(q)
        res = run_sparql(q)
        for b in res["results"]["bindings"]:
            qid = b["v"]["value"].rsplit("/", 1)[-1]
            out.setdefault(qid, []).append(b["alt"]["value"])
        time.sleep(1.0)
    return queries, out


def qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def fetch_value_meta(qids: list[str], batch: int = 300) -> tuple[list[str], dict[str, dict]]:
    """sitelinks count and English Wikipedia URL (if any) for answer entities."""
    out: dict[str, dict] = {}
    queries: list[str] = []
    for i in range(0, len(qids), batch):
        chunk = qids[i : i + batch]
        q = VALUE_META_QUERY.format(values=" ".join(f"wd:{x}" for x in chunk))
        queries.append(q)
        for b in run_sparql(q)["results"]["bindings"]:
            out[qid(b["v"]["value"])] = {
                "sitelinks": int(b["sitelinks"]["value"]),
                "enwiki": b.get("enwiki", {}).get("value"),
            }
        time.sleep(1.0)
    return queries, out


def fetch_reference_counts(
    statement_ids: list[str], batch: int = 300
) -> tuple[list[str], dict[str, int]]:
    """Number of Wikidata references attached to each statement (0 = unsourced)."""
    out: dict[str, int] = {}
    queries: list[str] = []
    for i in range(0, len(statement_ids), batch):
        chunk = statement_ids[i : i + batch]
        q = REF_COUNT_QUERY.format(values=" ".join(f"wds:{x}" for x in chunk))
        queries.append(q)
        for b in run_sparql(q)["results"]["bindings"]:
            out[qid(b["st"]["value"])] = int(b["refs"]["value"])
        time.sleep(1.0)
    return queries, out
