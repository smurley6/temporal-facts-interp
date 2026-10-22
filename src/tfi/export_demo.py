"""Export a compact JSON bundle for the website explorer (precomputed, no model in the browser).

    python -m tfi.export_demo --run results/qwen2.5-3b --out results/qwen2.5-3b/demo.json

Contents
  meta        model, cutoff, snapshot date, counts, headline metrics
  layers      per-layer probe AUROC / accuracy + controls (for the curve)
  items       every evaluated fact: prompt, truth, model answer, grade, class, per-layer
              P(error) trace (rounded), whether the probe flagged it
  pairs       patching pairs with their recovery grids (rounded)
  patching    mean recovery grids + peaks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/qwen2.5-3b")
    ap.add_argument("--out", default=None)
    ap.add_argument("--benchmark-manifest", default="data/processed/manifest.json")
    a = ap.parse_args()
    run = Path(a.run)
    out = Path(a.out) if a.out else run / "demo.json"

    rows = [json.loads(line) for line in open(run / "answers.jsonl")]
    run_manifest = json.loads((run / "run_manifest.json").read_text())
    bench_manifest = json.loads(Path(a.benchmark_manifest).read_text())
    probes = json.loads((run / "probes" / "summary.json").read_text())
    layer_df = pd.read_csv(run / "probes" / "layer_metrics.csv")
    oof = np.load(run / "probes" / "oof_probs.npy")  # [n, L]
    layer_star = probes["selected_layer_modal"]
    base_rate = probes["error_rate"]

    items = []
    for i, r in enumerate(rows):
        trace = oof[i]
        items.append(
            {
                "id": r["id"],
                "relation": r["relation"],
                "class": r["label"],
                "subject": r["subject_label"],
                "prompt": r["prompt"],
                "answer": r["answer"],
                "answer_since": r["answer_since"],
                "outdated": r["outdated"],
                "model_answer": r["continuation"].split("\n")[0].strip().rstrip("."),
                "grade": r["grade"],
                "p_error": [round(float(x), 3) for x in trace],
                "p_error_selected": round(float(trace[layer_star]), 3),
                "flagged": bool(trace[layer_star] >= base_rate),
                "sitelinks": r["sitelinks"],
            }
        )

    pairs, patching = [], None
    if (run / "patching" / "summary.json").exists():
        patching = json.loads((run / "patching" / "summary.json").read_text())
        rec = np.load(run / "patching" / "recovery.npy")  # [P, C, 2, L]
        infos = [json.loads(line) for line in open(run / "patching" / "pairs.jsonl")]
        for info, grid in zip(infos, rec, strict=True):
            pairs.append({**info, "recovery": np.round(grid, 3).tolist()})

    bundle = {
        "meta": {
            "model": run_manifest["model"],
            "cutoff": bench_manifest["cutoff"],
            "snapshot_date": bench_manifest["snapshot_date"],
            "n_items": len(items),
            "n_layers": int(oof.shape[1]),
            "grade_counts": run_manifest["grade_counts"],
            "selected_layer": layer_star,
            "headline": {
                "probe_auroc": probes["selected_layer_metrics"]["auroc"],
                "probe_accuracy": probes["selected_layer_metrics"][
                    "accuracy_at_base_rate_threshold"
                ],
                "majority_accuracy": probes["majority_accuracy"],
                "shuffled_auroc": probes["controls_at_peak_layer"]["shuffled_auroc"],
                "random_dir_auroc": probes["controls_at_peak_layer"]["random_dir_auroc"],
                "per_class": probes["per_class"],
                "ece": probes["calibration"]["ece"],
                "mlp_entity_mid_stack_share": patching["mlp_entity_mid_stack_share"]
                if patching
                else None,
                "n_pairs": patching["n_pairs"] if patching else 0,
            },
        },
        "layers": layer_df.round(4).to_dict(orient="records"),
        "items": items,
        "pairs": pairs,
        "patching": {
            k: patching[k]
            for k in ("mean_recovery", "peak", "components", "positions", "mid_stack_layers")
        }
        if patching
        else None,
    }
    out.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")))
    print(
        f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(items)} items, {len(pairs)} pairs)"
    )


if __name__ == "__main__":
    main()
