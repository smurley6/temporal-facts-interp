"""Run the model over the benchmark: answers, grades, last-token residual stream.

    python -m tfi.run_model --out results/qwen2.5-3b [--limit N]

Writes
  <out>/answers.jsonl      one row per evaluated item (demonstration items are excluded)
  <out>/activations.npy    float16 [n_items, n_layers, d_model], row order = answers.jsonl
  <out>/run_manifest.json  model, dtype, few-shot demos, timing
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from tfi.model import (
    DEFAULT_MODEL,
    build_fewshot_prefixes,
    first_token_id,
    full_prompt,
    grade,
    load_model,
    run_item,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="data/processed/benchmark.jsonl")
    ap.add_argument("--out", default="results/qwen2.5-3b")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    items = [json.loads(line) for line in open(a.benchmark)]
    prefixes, demo_ids = build_fewshot_prefixes(items)
    evaluation = [x for x in items if x["id"] not in demo_ids]
    if a.limit:
        # a stratified-ish slice: round-robin over labels so small runs still see every class
        by_label: dict[str, list[dict]] = {}
        for x in evaluation:
            by_label.setdefault(x["label"], []).append(x)
        sliced: list[dict] = []
        i = 0
        while len(sliced) < a.limit and any(by_label.values()):
            for lab in list(by_label):
                if by_label[lab] and len(sliced) < a.limit:
                    sliced.append(by_label[lab].pop(i % len(by_label[lab]) if by_label[lab] else 0))
            i += 7
        evaluation = sliced

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(a.model, device=a.device)
    t0 = time.time()
    rows: list[dict] = []
    acts: list[np.ndarray] = []
    with open(out / "answers.jsonl", "w") as f:
        for i, item in enumerate(evaluation):
            prompt = full_prompt(prefixes, item)
            ans_tok = first_token_id(model, item["answer"])
            out_tok = first_token_id(model, item["outdated"]) if item.get("outdated") else None
            r = run_item(model, prompt, ans_tok, out_tok)
            former = [a for f in item.get("former", []) for a in (f["label"], *f["aliases"])]
            g = grade(
                r.continuation,
                [item["answer"], *item["answer_aliases"]],
                [item.get("outdated") or "", *item.get("outdated_aliases", [])],
                former,
            )
            row = {
                "id": item["id"],
                "relation": item["relation"],
                "label": item["label"],
                "subject_qid": item["subject_qid"],
                "subject_label": item["subject_label"],
                "prompt": item["prompt"],
                "answer": item["answer"],
                "outdated": item.get("outdated"),
                "answer_since": item["answer_since"],
                "sitelinks": item["sitelinks"],
                "continuation": r.continuation,
                "grade": g,
                "top_token": r.top_token,
                "answer_tok": ans_tok,
                "outdated_tok": out_tok,
                "answer_logit": r.answer_logit,
                "answer_rank": r.answer_rank,
                "outdated_logit": r.outdated_logit,
                "outdated_rank": r.outdated_rank,
            }
            rows.append(row)
            acts.append(r.resid)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0 or i + 1 == len(evaluation):
                el = time.time() - t0
                rate = el / (i + 1)
                eta = rate * (len(evaluation) - i - 1) / 60
                print(f"{i + 1}/{len(evaluation)}  {rate:.2f}s/item  eta {eta:.1f} min", flush=True)
    np.save(out / "activations.npy", np.stack(acts))
    manifest = {
        "model": a.model,
        "dtype": "float16",
        "device": a.device,
        "n_items": len(rows),
        "fewshot_prefixes": prefixes,
        "demo_ids": sorted(demo_ids),
        "benchmark": a.benchmark,
        "started_at": datetime.fromtimestamp(t0, UTC).isoformat(timespec="seconds"),
        "seconds": round(time.time() - t0, 1),
        "grade_counts": {
            lab: {
                g: sum(1 for r in rows if r["label"] == lab and r["grade"] == g)
                for g in ("correct", "outdated", "stale", "other")
            }
            for lab in ("stable", "recent", "superseded")
        },
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest["grade_counts"], indent=2))


if __name__ == "__main__":
    main()
