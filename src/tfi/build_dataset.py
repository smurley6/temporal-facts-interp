"""Build the temporal-facts benchmark from Wikidata.

    python -m tfi.build_dataset --cutoff 2023-12-31 --out data/processed

Outputs
  data/raw/<relation>.<stamp>.json       frozen SPARQL responses (one per relation + aliases)
  data/processed/benchmark.jsonl         one labeled fact per line (stable / superseded / recent)
  data/processed/excluded.jsonl          every candidate that was dropped, with the reason
  data/processed/manifest.json           cutoff, snapshot date, counts, raw-file hashes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path

from tfi.labeling import LabelResult, Statement, label_fact
from tfi.relations import RELATIONS, Relation
from tfi.wikidata import (
    fetch_aliases,
    fetch_reference_counts,
    fetch_statements,
    fetch_value_meta,
    freeze,
    qid,
)

QID_RE = re.compile(r"^Q\d+$")


def _date(v: str | None) -> date | None:
    if not v:
        return None
    # Wikidata emits e.g. 2011-08-24T00:00:00Z; negative years are irrelevant here
    try:
        return date.fromisoformat(v[:10])
    except ValueError:
        return None


def _int(v: str | None) -> int | None:
    return int(v) if v else None


def parse_rows(bindings: list[dict]) -> dict[str, dict]:
    """Group SPARQL rows into {subject_qid: {label, sitelinks, office, statements[]}}."""
    subjects: dict[str, dict] = {}
    seen: set[str] = set()
    for b in bindings:
        s = qid(b["s"]["value"])
        st_id = qid(b["st"]["value"])
        key = f"{s}|{st_id}"
        if key in seen:  # duplicate rows from multiple office values etc.
            continue
        seen.add(key)
        entry = subjects.setdefault(
            s,
            {
                "subject_qid": s,
                "subject_label": b["sLabel"]["value"],
                "sitelinks": int(b["sitelinks"]["value"]),
                "office_label": b.get("officeLabel", {}).get("value"),
                "office_qid": qid(b["office"]["value"]) if "office" in b else None,
                "statements": [],
            },
        )
        rank = b["rank"]["value"].rsplit("#", 1)[-1].replace("Rank", "").lower()
        entry["statements"].append(
            Statement(
                statement_id=st_id,
                value_qid=qid(b["v"]["value"]),
                value_label=b["vLabel"]["value"],
                rank=rank,
                start=_date(b.get("start", {}).get("value")),
                start_precision=_int(b.get("startPrec", {}).get("value")),
                end=_date(b.get("end", {}).get("value")),
                end_precision=_int(b.get("endPrec", {}).get("value")),
            )
        )
    return subjects


def render_prompt(rel: Relation, subj: dict) -> str | None:
    if rel.office_prop:
        office = subj.get("office_label")
        if not office or QID_RE.match(office):
            return None
        return rel.template.format(office=office)
    return rel.template.format(subject=subj["subject_label"])


def _former(statements: list[Statement], current: Statement) -> list[dict]:
    """Every distinct past value of the fact (any rank but deprecated), oldest first."""
    seen: set[str] = set()
    out: list[dict] = []
    for st in sorted(statements, key=lambda x: x.start or date.min):
        if st.value_qid == current.value_qid or st.value_qid in seen or st.rank == "deprecated":
            continue
        if QID_RE.match(st.value_label):
            continue
        seen.add(st.value_qid)
        out.append(
            {
                "qid": st.value_qid,
                "label": st.value_label,
                "aliases": [],
                "start": st.start.isoformat() if st.start else None,
                "end": st.end.isoformat() if st.end else None,
            }
        )
    return out


def _st_dict(s: Statement | None) -> dict | None:
    if s is None:
        return None
    d = asdict(s)
    d["start"] = s.start.isoformat() if s.start else None
    d["end"] = s.end.isoformat() if s.end else None
    return d


def build(cutoff: date, out_dir: Path, raw_dir: Path, relations: list[str]) -> None:
    today = datetime.now(UTC).date()
    out_dir.mkdir(parents=True, exist_ok=True)
    kept: list[dict] = []
    excluded: list[dict] = []
    raw_files: list[Path] = []
    counts: dict[str, Counter] = defaultdict(Counter)

    for key in relations:
        rel = RELATIONS[key]
        t0 = time.time()
        query, res = fetch_statements(
            pid=rel.pid,
            min_sitelinks=rel.min_sitelinks,
            subject_filter=rel.subject_filter,
            office_prop=rel.office_prop,
        )
        raw_files.append(freeze(raw_dir, rel.key, query, res))
        subjects = parse_rows(res["results"]["bindings"])
        print(f"[{rel.key}] {len(subjects)} subjects in {time.time() - t0:.1f}s", flush=True)
        # English Wikipedia article required for the subject (popularity floor the 3B model can
        # plausibly clear); checked in cheap batched lookups rather than inside the main query,
        # which times out on WDQS when the join is included there.
        subj_queries, smeta = fetch_value_meta(sorted(subjects))
        raw_files.append(
            freeze(
                raw_dir,
                f"{rel.key}.subject_meta",
                "\n\n".join(subj_queries),
                {"subject_meta": smeta},
            )
        )
        for s in subjects.values():
            s["enwiki"] = smeta.get(s["subject_qid"], {}).get("enwiki")
        for s in subjects.values():
            if not s.get("enwiki"):
                excluded.append(
                    {
                        "relation": rel.key,
                        "subject_qid": s["subject_qid"],
                        "reason": "subject_without_enwiki",
                    }
                )
                counts[rel.key]["excluded:subject_without_enwiki"] += 1
                continue
            if QID_RE.match(s["subject_label"]):
                excluded.append(
                    {
                        "relation": rel.key,
                        "subject_qid": s["subject_qid"],
                        "reason": "no_english_label",
                    }
                )
                counts[rel.key]["excluded:no_english_label"] += 1
                continue
            prompt = render_prompt(rel, s)
            if prompt is None:
                excluded.append(
                    {
                        "relation": rel.key,
                        "subject_qid": s["subject_qid"],
                        "reason": "no_office_label",
                    }
                )
                counts[rel.key]["excluded:no_office_label"] += 1
                continue
            r: LabelResult = label_fact(s["statements"], cutoff=cutoff, today=today)
            if r.label == "excluded":
                excluded.append(
                    {
                        "relation": rel.key,
                        "subject_qid": s["subject_qid"],
                        "subject_label": s["subject_label"],
                        "reason": r.reason,
                    }
                )
                counts[rel.key][f"excluded:{r.reason}"] += 1
                continue
            if QID_RE.match(r.current.value_label) or (
                r.previous and QID_RE.match(r.previous.value_label)
            ):
                excluded.append(
                    {
                        "relation": rel.key,
                        "subject_qid": s["subject_qid"],
                        "reason": "value_without_english_label",
                    }
                )
                counts[rel.key]["excluded:value_without_english_label"] += 1
                continue
            counts[rel.key][r.label] += 1
            kept.append(
                {
                    "id": f"{rel.key}:{s['subject_qid']}",
                    "relation": rel.key,
                    "property": rel.pid,
                    "subject_qid": s["subject_qid"],
                    "subject_label": s["subject_label"],
                    "sitelinks": s["sitelinks"],
                    "subject_enwiki": s["enwiki"],
                    "office_qid": s["office_qid"],
                    "office_label": s["office_label"],
                    "prompt": prompt,
                    "label": r.label,
                    "answer_qid": r.current.value_qid,
                    "answer": r.current.value_label,
                    "answer_aliases": [],
                    "answer_since": r.current.start.isoformat(),
                    "outdated_qid": r.previous.value_qid if r.previous else None,
                    "outdated": r.previous.value_label if r.previous else None,
                    "outdated_aliases": [],
                    "outdated_until": r.previous.end.isoformat()
                    if r.previous and r.previous.end
                    else None,
                    "former": _former(s["statements"], r.current),
                    "current_statement": _st_dict(r.current),
                    "previous_statement": _st_dict(r.previous),
                    "n_statements": len(s["statements"]),
                }
            )
        time.sleep(2.0)

    # aliases for every answer / outdated value
    value_qids = sorted(
        {x["answer_qid"] for x in kept}
        | {x["outdated_qid"] for x in kept if x["outdated_qid"]}
        | {f["qid"] for x in kept for f in x["former"]}
    )
    print(f"fetching aliases for {len(value_qids)} values")
    alias_queries, aliases = fetch_aliases(value_qids)
    raw_files.append(freeze(raw_dir, "aliases", "\n\n".join(alias_queries), {"aliases": aliases}))
    meta_queries, vmeta = fetch_value_meta(value_qids)
    raw_files.append(
        freeze(raw_dir, "value_meta", "\n\n".join(meta_queries), {"value_meta": vmeta})
    )
    st_ids = sorted(
        {x["current_statement"]["statement_id"] for x in kept}
        | {x["previous_statement"]["statement_id"] for x in kept if x["previous_statement"]}
    )
    print(f"fetching reference counts for {len(st_ids)} statements")
    ref_queries, refs = fetch_reference_counts(st_ids)
    raw_files.append(freeze(raw_dir, "statement_refs", "\n\n".join(ref_queries), {"refs": refs}))
    filtered: list[dict] = []
    for x in kept:
        x["answer_aliases"] = sorted(set(aliases.get(x["answer_qid"], [])))
        for fm in x["former"]:
            fm["aliases"] = sorted(set(aliases.get(fm["qid"], [])))
        am = vmeta.get(x["answer_qid"], {})
        x["answer_sitelinks"] = am.get("sitelinks")
        x["answer_enwiki"] = am.get("enwiki")
        x["answer_refs"] = refs.get(x["current_statement"]["statement_id"], 0)
        if x["outdated_qid"]:
            x["outdated_aliases"] = sorted(set(aliases.get(x["outdated_qid"], [])))
            om = vmeta.get(x["outdated_qid"], {})
            x["outdated_sitelinks"] = om.get("sitelinks")
            x["outdated_enwiki"] = om.get("enwiki")
            x["outdated_refs"] = refs.get(x["previous_statement"]["statement_id"], 0)
        if not x["answer_enwiki"] or (x["outdated_qid"] and not x["outdated_enwiki"]):
            excluded.append(
                {
                    "relation": x["relation"],
                    "subject_qid": x["subject_qid"],
                    "subject_label": x["subject_label"],
                    "reason": "answer_without_enwiki",
                }
            )
            counts[x["relation"]][x["label"]] -= 1
            counts[x["relation"]]["excluded:answer_without_enwiki"] += 1
            continue
        filtered.append(x)
    kept = filtered

    kept.sort(key=lambda x: (x["relation"], -x["sitelinks"], x["subject_qid"]))
    (out_dir / "benchmark.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in kept)
    )
    (out_dir / "excluded.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in excluded)
    )
    manifest = {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "snapshot_date": today.isoformat(),
        "cutoff": cutoff.isoformat(),
        "relations": relations,
        "n_items": len(kept),
        "n_excluded": len(excluded),
        "by_label": dict(Counter(x["label"] for x in kept)),
        "by_relation": {k: dict(v) for k, v in counts.items()},
        "raw_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw_files},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("n_items", "n_excluded", "by_label")}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2023-12-31", help="model training cutoff (ISO date)")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--relations", nargs="*", default=list(RELATIONS))
    a = ap.parse_args()
    build(date.fromisoformat(a.cutoff), Path(a.out), Path(a.raw), a.relations)


if __name__ == "__main__":
    main()
