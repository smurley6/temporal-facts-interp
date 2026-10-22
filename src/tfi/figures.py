"""Figures from the result artifacts (no model needed).

  python -m tfi.figures --run results/qwen2.5-3b

fig_probe_layers.png      per-layer probe AUROC / accuracy with shuffled-label, random-direction
                          and majority controls
fig_patching.png          recovery heatmaps, 3 components x 2 positions
fig_calibration.png       reliability diagram + miss rate by shelf-life class
fig_cutoff.png            model accuracy by the year the current answer took office
                          (a data-driven check on the assumed training cutoff)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CLASS_ORDER = ("stable", "recent", "superseded")
CLASS_COLORS = {"stable": "#2a7f62", "recent": "#c99a2e", "superseded": "#b23a48"}


def probe_layers(run: Path, out: Path) -> None:
    df = pd.read_csv(run / "probes" / "layer_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ax = axes[0]
    ax.plot(df.layer, df.auroc, lw=2, color="#1f4e79", label="probe (entity-disjoint OOF)")
    ax.plot(df.layer, df.shuffled_auroc, lw=1.5, ls="--", color="#888", label="shuffled labels")
    ax.plot(df.layer, df.random_dir_auroc, lw=1.5, ls=":", color="#888", label="random direction")
    ax.axhline(0.5, color="#bbb", lw=1)
    ax.set(xlabel="layer", ylabel="AUROC", title="Does the residual stream predict an error?")
    ax.set_ylim(0.4, 1.0)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    ax.plot(df.layer, df.accuracy, lw=2, color="#1f4e79", label="probe")
    ax.plot(df.layer, df.shuffled_accuracy, lw=1.5, ls="--", color="#888", label="shuffled labels")
    ax.plot(
        df.layer, df.random_dir_accuracy, lw=1.5, ls=":", color="#888", label="random direction"
    )
    ax.axhline(df.majority_accuracy.iloc[0], color="#bbb", lw=1, label="majority class")
    ax.set(xlabel="layer", ylabel="accuracy", title="Accuracy (threshold 0.5)")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out / "fig_probe_layers.png", dpi=160)
    plt.close(fig)


def patching(run: Path, out: Path) -> None:
    s = json.loads((run / "patching" / "summary.json").read_text())
    comps, poss = s["components"], s["positions"]
    fig, axes = plt.subplots(1, len(poss), figsize=(12, 3.6), constrained_layout=True)
    pretty = {
        "hook_resid_post": "residual",
        "hook_attn_out": "attention out",
        "hook_mlp_out": "MLP out",
    }
    for pi, pos in enumerate(poss):
        grid = np.array([s["mean_recovery"][c][pos] for c in comps])
        ax = axes[pi]
        im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_yticks(range(len(comps)), [pretty[c] for c in comps])
        ax.set_xlabel("layer")
        ax.set_title(f"patch at {pos.replace('_', ' ')} token  (n={s['n_pairs']} pairs)")
    fig.colorbar(im, ax=axes, label="fraction of logit difference recovered", shrink=0.9)
    fig.savefig(out / "fig_patching.png", dpi=160)
    plt.close(fig)


def calibration(run: Path, out: Path) -> None:
    s = json.loads((run / "probes" / "summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ax = axes[0]
    bins = [b for b in s["calibration"]["bins"] if b["n"]]
    ax.plot([0, 1], [0, 1], color="#bbb", lw=1)
    ax.plot([b["conf"] for b in bins], [b["acc"] for b in bins], marker="o", color="#1f4e79")
    for b in bins:
        ax.annotate(
            str(b["n"]),
            (b["conf"], b["acc"]),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.set(
        xlabel="probe P(error)",
        ylabel="observed error rate",
        title=f"Reliability (ECE {s['calibration']['ece']:.3f})",
    )
    ax = axes[1]
    pc = s["per_class"]
    xs = np.arange(len(CLASS_ORDER))
    miss = [pc[c]["miss_rate"] or 0 for c in CLASS_ORDER]
    err = [pc[c]["error_rate"] or 0 for c in CLASS_ORDER]
    ax.bar(
        xs - 0.2,
        err,
        0.4,
        color=[CLASS_COLORS[c] for c in CLASS_ORDER],
        alpha=0.45,
        label="model error rate",
    )
    ax.bar(
        xs + 0.2,
        miss,
        0.4,
        color=[CLASS_COLORS[c] for c in CLASS_ORDER],
        label="errors the probe missed",
    )
    ax.set_xticks(xs, [f"{c}\n(n={pc[c]['n']})" for c in CLASS_ORDER])
    ax.set_ylim(0, 1)
    ax.set_title("Where the internal error signal fails")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out / "fig_calibration.png", dpi=160)
    plt.close(fig)


def cutoff(run: Path, out: Path, cutoff_date: str | None) -> None:
    rows = [json.loads(line) for line in open(run / "answers.jsonl")]
    by_year: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        y = int(r["answer_since"][:4])
        if y >= 2000:
            by_year[y].append(1 if r["grade"] == "correct" else 0)
    years = sorted(by_year)
    acc = [np.mean(by_year[y]) for y in years]
    n = [len(by_year[y]) for y in years]
    fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
    ax.bar(years, acc, color="#1f4e79")
    for x, a, k in zip(years, acc, n, strict=True):
        ax.annotate(
            str(k), (x, a), ha="center", fontsize=7, xytext=(0, 2), textcoords="offset points"
        )
    if cutoff_date:
        ax.axvline(
            int(cutoff_date[:4]) + 0.5,
            color="#b23a48",
            ls="--",
            lw=1.2,
            label=f"assumed cutoff {cutoff_date}",
        )
        ax.legend(frameon=False, fontsize=8)
    ax.set(
        xlabel="year the current answer took office",
        ylabel="model accuracy",
        ylim=(0, 1),
        title="Knowledge falls off a cliff after the training cutoff",
    )
    fig.savefig(out / "fig_cutoff.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/qwen2.5-3b")
    ap.add_argument("--benchmark-manifest", default="data/processed/manifest.json")
    a = ap.parse_args()
    run = Path(a.run)
    out = run / "figures"
    out.mkdir(exist_ok=True)
    cutoff_date = None
    if Path(a.benchmark_manifest).exists():
        cutoff_date = json.loads(Path(a.benchmark_manifest).read_text()).get("cutoff")
    cutoff(run, out, cutoff_date)
    if (run / "probes" / "layer_metrics.csv").exists():
        probe_layers(run, out)
        calibration(run, out)
    if (run / "patching" / "summary.json").exists():
        patching(run, out)
    print("figures written to", out)


if __name__ == "__main__":
    main()
