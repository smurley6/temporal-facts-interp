"""Per-layer linear probes: does the residual stream predict whether the answer will be wrong?

    python -m tfi.probes --run results/qwen2.5-3b

Design
  target        y = 1 if the model's greedy answer is NOT correct (an error)
  features      residual stream after block L at the last prompt token, standardized on train
  splits        5-fold GroupKFold by subject entity (no entity appears in both train and test)
  regularizer   C chosen per layer by inner GroupKFold on the training fold only
  controls      shuffled labels (train on permuted y, test on true y)
                random direction (1-D logistic on the projection onto a random unit vector, x20)
                majority class
  selection     the layer used for calibration / per-class analysis is chosen by inner-fold
                AUROC inside each outer fold (nested), never on the test fold
  extras        relation-held-out transfer at the selected layer; reliability bins and
                miss rate (errors the probe called "correct") per shelf-life class

Outputs (<run>/probes/)
  layer_metrics.csv   per-layer accuracy / balanced accuracy / AUROC for probe and controls
  oof_probs.npy       out-of-fold P(error) for every item x layer, row order = answers.jsonl
  summary.json        selected layers, per-class miss rates, calibration, transfer table
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

C_GRID = (0.003, 0.03, 0.3)
N_RANDOM_DIRS = 20
N_BINS = 10


def fit_probe(x: np.ndarray, y: np.ndarray, c: float) -> tuple[StandardScaler, LogisticRegression]:
    sc = StandardScaler().fit(x)
    clf = LogisticRegression(C=c, max_iter=3000).fit(sc.transform(x), y)
    return sc, clf


def predict(sc: StandardScaler, clf: LogisticRegression, x: np.ndarray) -> np.ndarray:
    return clf.predict_proba(sc.transform(x))[:, 1]


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def inner_select_c(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int
) -> tuple[float, float]:
    """Return (best C, its inner AUROC) using GroupKFold(3) on the training fold."""
    best = (C_GRID[0], -1.0)
    for c in C_GRID:
        aucs = []
        for tr, va in GroupKFold(n_splits=3).split(x, y, groups):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                continue
            sc, clf = fit_probe(x[tr], y[tr], c)
            aucs.append(safe_auc(y[va], predict(sc, clf, x[va])))
        m = float(np.nanmean(aucs)) if aucs else -1.0
        if m > best[1]:
            best = (c, m)
    return best


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> tuple[float, list[dict]]:
    edges = np.linspace(0, 1, n_bins + 1)
    bins, total = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            bins.append({"lo": lo, "hi": hi, "n": 0, "conf": None, "acc": None})
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        total += m.sum() / len(p) * abs(conf - acc)
        bins.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "conf": conf, "acc": acc})
    return float(total), bins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/qwen2.5-3b")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--subset",
        choices=("all", "known"),
        default="all",
        help="known = only items where the model named a real past or present holder "
        "(grade correct / outdated / stale); asks whether the model knows its knowledge is stale "
        "rather than whether it knows the entity at all",
    )
    a = ap.parse_args()
    run = Path(a.run)
    out = run / ("probes" if a.subset == "all" else f"probes_{a.subset}")
    out.mkdir(exist_ok=True)

    rows = [json.loads(line) for line in open(run / "answers.jsonl")]
    acts = np.load(run / "activations.npy").astype(np.float32)  # [n, L, d]
    if a.subset == "known":
        keep = [i for i, r in enumerate(rows) if r["grade"] in ("correct", "outdated", "stale")]
        rows = [rows[i] for i in keep]
        acts = acts[keep]
        (out / "row_index.json").write_text(json.dumps(keep))
    n, n_layers, _ = acts.shape
    y = np.array([0 if r["grade"] == "correct" else 1 for r in rows])
    groups = np.array([r["subject_qid"] for r in rows])
    labels = np.array([r["label"] for r in rows])
    relations = np.array([r["relation"] for r in rows])
    rng = np.random.default_rng(a.seed)
    print(f"n={n} layers={n_layers} error rate={y.mean():.3f}")

    outer = list(GroupKFold(n_splits=a.folds).split(acts[:, 0], y, groups))
    oof = np.full((n, n_layers), np.nan)
    oof_shuf = np.full((n, n_layers), np.nan)
    inner_auc = np.zeros((a.folds, n_layers))
    chosen_c = np.zeros((a.folds, n_layers))
    rand_auc = np.zeros((a.folds, n_layers))
    rand_acc = np.zeros((a.folds, n_layers))
    dirs = rng.standard_normal((N_RANDOM_DIRS, acts.shape[2])).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    for k, (tr, te) in enumerate(outer):
        for layer in range(n_layers):
            x = acts[:, layer]
            c, ia = inner_select_c(x[tr], y[tr], groups[tr], a.seed)
            inner_auc[k, layer], chosen_c[k, layer] = ia, c
            sc, clf = fit_probe(x[tr], y[tr], c)
            oof[te, layer] = predict(sc, clf, x[te])
            # shuffled-label control
            y_perm = rng.permutation(y[tr])
            sc_s, clf_s = fit_probe(x[tr], y_perm, c)
            oof_shuf[te, layer] = predict(sc_s, clf_s, x[te])
            # random-direction control
            aucs, accs = [], []
            for d in dirs:
                z_tr, z_te = (x[tr] @ d)[:, None], (x[te] @ d)[:, None]
                sc_r, clf_r = fit_probe(z_tr, y[tr], 1.0)
                p = predict(sc_r, clf_r, z_te)
                aucs.append(safe_auc(y[te], p))
                accs.append(float(((p >= 0.5) == y[te]).mean()))
            rand_auc[k, layer], rand_acc[k, layer] = float(np.nanmean(aucs)), float(np.mean(accs))
        print(
            f"fold {k + 1}/{a.folds} done; best inner layer {int(inner_auc[k].argmax())}",
            flush=True,
        )

    majority = float(max(y.mean(), 1 - y.mean()))
    metrics = []
    for layer in range(n_layers):
        p, ps = oof[:, layer], oof_shuf[:, layer]
        metrics.append(
            {
                "layer": layer,
                "accuracy": float(((p >= 0.5) == y).mean()),
                "balanced_accuracy": float(balanced_accuracy_score(y, p >= 0.5)),
                "auroc": safe_auc(y, p),
                "shuffled_accuracy": float(((ps >= 0.5) == y).mean()),
                "shuffled_auroc": safe_auc(y, ps),
                "random_dir_accuracy": float(rand_acc[:, layer].mean()),
                "random_dir_auroc": float(rand_auc[:, layer].mean()),
                "majority_accuracy": majority,
                "mean_inner_auroc": float(inner_auc[:, layer].mean()),
            }
        )
    df = pd.DataFrame(metrics)
    df.to_csv(out / "layer_metrics.csv", index=False)
    np.save(out / "oof_probs.npy", oof.astype(np.float32))

    # nested layer selection: per outer fold, the layer with the best inner AUROC
    sel_layers = inner_auc.argmax(axis=1)
    p_sel = np.full(n, np.nan)
    for k, (_, te) in enumerate(outer):
        p_sel[te] = oof[te, sel_layers[k]]
    # decision threshold: balanced-accuracy-maximizing on each training fold's inner OOF is
    # expensive; use the error base rate on the training fold (a calibrated probe's natural cut)
    thr = np.full(n, np.nan)
    for tr, te in outer:
        thr[te] = y[tr].mean()
    flagged = p_sel >= thr

    per_class = {}
    for lab in ("stable", "recent", "superseded"):
        m = labels == lab
        err = y[m] == 1
        per_class[lab] = {
            "n": int(m.sum()),
            "error_rate": float(err.mean()) if m.sum() else None,
            "auroc": safe_auc(y[m], p_sel[m]) if m.sum() else None,
            "miss_rate": float((~flagged[m][err]).mean()) if err.sum() else None,
            "false_alarm_rate": float(flagged[m][~err].mean()) if (~err).sum() else None,
            "mean_p_error": float(p_sel[m].mean()) if m.sum() else None,
            "ece": ece(y[m], p_sel[m])[0] if m.sum() else None,
        }
    overall_ece, bins = ece(y, p_sel)

    # relation-held-out transfer at the modal selected layer
    layer_star = int(np.bincount(sel_layers).argmax())
    transfer = {}
    for rel in sorted(set(relations)):
        tr, te = np.where(relations != rel)[0], np.where(relations == rel)[0]
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            transfer[rel] = {"n": int(len(te)), "auroc": None}
            continue
        c = float(np.median(chosen_c[:, layer_star]))
        sc, clf = fit_probe(acts[tr, layer_star], y[tr], c)
        p = predict(sc, clf, acts[te, layer_star])
        transfer[rel] = {
            "n": int(len(te)),
            "auroc": safe_auc(y[te], p),
            "accuracy": float(((p >= y[tr].mean()) == y[te]).mean()),
            "majority": float(max(y[te].mean(), 1 - y[te].mean())),
        }

    best = df.loc[df["auroc"].idxmax()]
    summary = {
        "n": int(n),
        "error_rate": float(y.mean()),
        "majority_accuracy": majority,
        "selected_layer_per_fold": sel_layers.tolist(),
        "selected_layer_modal": layer_star,
        "selected_layer_metrics": {
            "accuracy": float(((p_sel >= 0.5) == y).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y, p_sel >= 0.5)),
            "auroc": safe_auc(y, p_sel),
            "accuracy_at_base_rate_threshold": float((flagged == y).mean()),
        },
        "peak_layer_oof": {
            "layer": int(best["layer"]),
            "accuracy": float(best["accuracy"]),
            "auroc": float(best["auroc"]),
        },
        "controls_at_peak_layer": {
            "shuffled_accuracy": float(best["shuffled_accuracy"]),
            "shuffled_auroc": float(best["shuffled_auroc"]),
            "random_dir_accuracy": float(best["random_dir_accuracy"]),
            "random_dir_auroc": float(best["random_dir_auroc"]),
        },
        "per_class": per_class,
        "calibration": {"ece": overall_ece, "bins": bins},
        "relation_transfer": transfer,
        "c_grid": C_GRID,
        "folds": a.folds,
        "subset": a.subset,
        "grade_counts": {
            g: int(sum(1 for r in rows if r["grade"] == g))
            for g in ("correct", "outdated", "stale", "other")
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("selected_layer_modal", "selected_layer_metrics", "per_class")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
