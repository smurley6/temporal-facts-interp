"""Activation patching over entity-swap pairs: where is the fact looked up?

    python -m tfi.patching --run results/qwen2.5-3b --pairs 120

For a pair (A, B) of same-relation facts the model answers correctly, run the B prompt
(corrupted) while copying one activation from the A run (clean) into it, and measure how much
of the A-vs-B answer logit difference comes back:

    recovery = (LD_patched - LD_corrupted) / (LD_clean - LD_corrupted)
    LD       = logit(first token of A's answer) - logit(first token of B's answer)

Patched components: residual stream (hook_resid_post), attention output (hook_attn_out) and
MLP output (hook_mlp_out), at every layer, at two positions: the last token of the subject
entity and the final prompt token. Both prompts share the same suffix, so the last token aligns
at -1; the entity position is located per prompt by tokenizing up to the end of the subject.

Outputs (<run>/patching/)
  pairs.jsonl     the pairs used, with clean / corrupted logit differences
  recovery.npy    float32 [n_pairs, 3 components, 2 positions, n_layers]
  summary.json    mean recovery grids, peak layers, mid-stack MLP share at the entity token
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from tfi.model import DEFAULT_MODEL, load_model

COMPONENTS = ("hook_resid_post", "hook_attn_out", "hook_mlp_out")
POSITIONS = ("entity_last", "last")
WINDOW = (
    5  # sliding window of adjacent layers for the windowed attn / MLP patch at the entity token
)


def entity_last_position(model, full_prompt: str, subject: str) -> int | None:
    """Index of the last token of `subject` inside the tokenization of `full_prompt`."""
    idx = full_prompt.rfind(subject)
    if idx < 0:
        return None
    head = full_prompt[: idx + len(subject)]
    toks_head = model.to_tokens(head)[0]
    toks_full = model.to_tokens(full_prompt)[0]
    k = len(toks_head)
    if k > len(toks_full) or not torch.equal(toks_head, toks_full[:k]):
        return None  # tokenization boundary does not align with the subject span; skip pair
    return k - 1


def select_pairs(rows: list[dict], n_pairs: int, seed: int) -> list[tuple[dict, dict]]:
    """Same-relation pairs of correctly answered items whose answers start with different tokens.

    Stratified round-robin across relations so no single relation dominates.
    """
    rng = random.Random(seed)
    ok = [r for r in rows if r["grade"] == "correct" and r["answer_rank"] == 0]
    by_rel: dict[str, list[dict]] = {}
    for r in ok:
        by_rel.setdefault(r["relation"], []).append(r)
    pools: dict[str, list[tuple[dict, dict]]] = {}
    for rel, items in by_rel.items():
        rng.shuffle(items)
        cands = []
        for i in range(0, len(items) - 1, 2):
            a, b = items[i], items[i + 1]
            if a["answer_tok"] != b["answer_tok"] and a["subject_qid"] != b["subject_qid"]:
                cands.append((a, b))
        pools[rel] = cands
    pairs: list[tuple[dict, dict]] = []
    while len(pairs) < n_pairs and any(pools.values()):
        for rel in sorted(pools):
            if pools[rel] and len(pairs) < n_pairs:
                pairs.append(pools[rel].pop())
    return pairs


@torch.no_grad()
def logit_diff(model, prompt: str, tok_a: int, tok_b: int) -> float:
    logits = model(prompt)
    last = logits[0, -1].float()
    return float(last[tok_a] - last[tok_b])


@torch.no_grad()
def patch_pair(model, prefix: str, a: dict, b: dict) -> tuple[np.ndarray, dict] | None:
    clean, corrupt = prefix + a["prompt"], prefix + b["prompt"]
    subj_a = a.get("office_label") or a["subject_label"]
    subj_b = b.get("office_label") or b["subject_label"]
    pos_a = entity_last_position(
        model, clean, a["prompt_subject"] if "prompt_subject" in a else subj_a
    )
    pos_b = entity_last_position(
        model, corrupt, b["prompt_subject"] if "prompt_subject" in b else subj_b
    )
    if pos_a is None or pos_b is None:
        return None
    tok_a, tok_b = a["answer_tok"], b["answer_tok"]
    n_layers = model.cfg.n_layers
    _, cache = model.run_with_cache(clean, names_filter=lambda n: n.endswith(COMPONENTS))
    ld_clean = logit_diff(model, clean, tok_a, tok_b)
    ld_corr = logit_diff(model, corrupt, tok_a, tok_b)
    denom = ld_clean - ld_corr
    if denom <= 0.5:  # the pair does not discriminate; skip
        return None
    rec = np.zeros((len(COMPONENTS), len(POSITIONS), n_layers), dtype=np.float32)
    positions = {"entity_last": (pos_a, pos_b), "last": (-1, -1)}

    def make_hook(name: str, pa: int, pb: int):
        src = cache[name][0, pa]

        def hook(act, hook):
            act[:, pb, :] = src.to(act.dtype)
            return act

        return hook

    def recovery(hooks: list) -> float:
        logits = model.run_with_hooks(corrupt, fwd_hooks=hooks)
        last = logits[0, -1].float()
        return (float(last[tok_a] - last[tok_b]) - ld_corr) / denom

    for ci, comp in enumerate(COMPONENTS):
        for pi, pname in enumerate(POSITIONS):
            pa, pb = positions[pname]
            for layer in range(n_layers):
                name = f"blocks.{layer}.{comp}"
                rec[ci, pi, layer] = recovery([(name, make_hook(name, pa, pb))])
    # windowed patch: attn / MLP outputs of WINDOW adjacent layers centred on each layer, at the
    # entity token (the measure causal-tracing papers use for "where is the fact looked up")
    win = np.zeros((2, n_layers), dtype=np.float32)
    for wi, comp in enumerate(("hook_attn_out", "hook_mlp_out")):
        for layer in range(n_layers):
            lo, hi = max(0, layer - WINDOW // 2), min(n_layers, layer + WINDOW // 2 + 1)
            hooks = [
                (f"blocks.{k}.{comp}", make_hook(f"blocks.{k}.{comp}", pos_a, pos_b))
                for k in range(lo, hi)
            ]
            win[wi, layer] = recovery(hooks)
    info = {
        "a_id": a["id"],
        "b_id": b["id"],
        "relation": a["relation"],
        "a_subject": a["subject_label"],
        "b_subject": b["subject_label"],
        "a_answer": a["answer"],
        "b_answer": b["answer"],
        "ld_clean": ld_clean,
        "ld_corrupt": ld_corr,
        "entity_pos_clean": int(pos_a),
        "entity_pos_corrupt": int(pos_b),
        "n_tokens_clean": int(model.to_tokens(clean).shape[1]),
    }
    return np.concatenate([rec.reshape(-1, n_layers), win]).reshape(-1, n_layers), info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/qwen2.5-3b")
    ap.add_argument("--pairs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    run = Path(a.run)
    out = run / "patching"
    out.mkdir(exist_ok=True)
    rows = [json.loads(line) for line in open(run / "answers.jsonl")]
    manifest = json.loads((run / "run_manifest.json").read_text())
    prefixes = manifest["fewshot_prefixes"]
    bench_rows = [json.loads(line) for line in open(manifest["benchmark"])]
    bench = {x["id"]: x for x in bench_rows}
    for r in rows:  # subject string as it appears in the prompt (office label for P6/P35)
        it = bench.get(r["id"], {})
        r["prompt_subject"] = it.get("office_label") or r["subject_label"]

    pairs = select_pairs(rows, a.pairs * 2, a.seed)  # oversample; some pairs are skipped
    model = load_model(a.model, device=a.device)
    recs, infos = [], []
    for i, (pa, pb) in enumerate(pairs):
        if len(recs) >= a.pairs:
            break
        res = patch_pair(model, prefixes[pa["relation"]], pa, pb)
        if res is None:
            continue
        rec, info = res
        recs.append(rec)
        infos.append(info)
        if len(recs) % 10 == 0:
            print(f"{len(recs)} pairs done ({i + 1} tried)", flush=True)
    flat = np.stack(recs)  # [P, 3*2 + 2, L]
    n_layers = flat.shape[-1]
    R = flat[:, : len(COMPONENTS) * len(POSITIONS)].reshape(
        -1, len(COMPONENTS), len(POSITIONS), n_layers
    )
    W = flat[:, len(COMPONENTS) * len(POSITIONS) :]  # [P, 2, L]: windowed attn, windowed mlp
    np.save(out / "recovery.npy", R)
    np.save(out / "recovery_windowed.npy", W)
    with open(out / "pairs.jsonl", "w") as f:
        for info in infos:
            f.write(json.dumps(info, ensure_ascii=False) + "\n")
    third = n_layers // 3
    mid = slice(third, 2 * third)
    mean = R.mean(axis=0)  # [C, 2, L]
    wmean = W.mean(axis=0)  # [2, L]
    mlp_entity = np.clip(mean[2, 0], 0, None)
    mlp_entity_win = np.clip(wmean[1], 0, None)
    attn_last = np.clip(mean[1, 1], 0, None)
    summary = {
        "n_pairs": int(R.shape[0]),
        "components": list(COMPONENTS),
        "positions": list(POSITIONS),
        "n_layers": int(n_layers),
        "mid_stack_layers": [third, 2 * third - 1],
        "mean_recovery": {
            comp: {pos: mean[ci, pi].round(4).tolist() for pi, pos in enumerate(POSITIONS)}
            for ci, comp in enumerate(COMPONENTS)
        },
        "peak": {
            comp: {
                pos: {"layer": int(mean[ci, pi].argmax()), "recovery": float(mean[ci, pi].max())}
                for pi, pos in enumerate(POSITIONS)
            }
            for ci, comp in enumerate(COMPONENTS)
        },
        "window": WINDOW,
        "windowed_mean_recovery": {
            "hook_attn_out": wmean[0].round(4).tolist(),
            "hook_mlp_out": wmean[1].round(4).tolist(),
        },
        "windowed_peak": {
            "hook_attn_out": {"layer": int(wmean[0].argmax()), "recovery": float(wmean[0].max())},
            "hook_mlp_out": {"layer": int(wmean[1].argmax()), "recovery": float(wmean[1].max())},
        },
        "mlp_entity_mid_stack_share": float(mlp_entity[mid].sum() / max(mlp_entity.sum(), 1e-9)),
        "mlp_entity_windowed_mid_stack_share": float(
            mlp_entity_win[mid].sum() / max(mlp_entity_win.sum(), 1e-9)
        ),
        "attn_last_mid_stack_share": float(attn_last[mid].sum() / max(attn_last.sum(), 1e-9)),
        "resid_entity_max_recovery": float(mean[0, 0].max()),
        "resid_last_max_recovery": float(mean[0, 1].max()),
        "by_relation": {
            rel: {
                "n": int(sum(1 for x in infos if x["relation"] == rel)),
                "mlp_entity_peak_layer": int(
                    R[[x["relation"] == rel for x in infos]].mean(axis=0)[2, 0].argmax()
                ),
            }
            for rel in sorted({x["relation"] for x in infos})
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    keys = (
        "n_pairs",
        "peak",
        "windowed_peak",
        "mlp_entity_mid_stack_share",
        "mlp_entity_windowed_mid_stack_share",
    )
    print(json.dumps({k: summary[k] for k in keys}, indent=2))


if __name__ == "__main__":
    main()
