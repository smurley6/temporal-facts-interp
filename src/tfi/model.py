"""Model access: loading, few-shot prompting, answer grading, activation caching.

Only this module touches the model. Everything downstream consumes the JSONL + NumPy
artifacts it writes, so probes / patching / figures re-run without a GPU or a model.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import torch

DEFAULT_MODEL = "Qwen/Qwen2.5-3B"
N_SHOTS = 3


def load_model(name: str = DEFAULT_MODEL, device: str = "mps", dtype: torch.dtype = torch.float16):
    from transformer_lens import TransformerBridge

    model = TransformerBridge.boot_transformers(name, device=device, dtype=dtype)
    model.eval()
    return model


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def grade(
    continuation: str, answers: list[str], outdated: list[str], former: list[str] = ()
) -> str:
    """correct | outdated | stale | other, by alias containment in the normalized continuation.

    outdated = the value that held at the training cutoff (superseded facts only);
    stale    = any other former holder of the fact (an even older answer).
    Only the first clause of the continuation counts (up to a newline or period), so a model
    that rambles into the right name three sentences later gets no credit.
    """
    first = re.split(r"[\n.]", continuation, maxsplit=1)[0]
    text = f" {normalize(first)} "

    def hit(cands: list[str]) -> bool:
        return any(f" {normalize(a)} " in text for a in cands if normalize(a))

    if hit(answers):
        return "correct"
    if hit(outdated):
        return "outdated"
    if hit(list(former)):
        return "stale"
    return "other"


def build_fewshot_prefixes(
    items: list[dict], n_shots: int = N_SHOTS
) -> tuple[dict[str, str], set[str]]:
    """Per relation, the n most popular stable facts are the demonstrations; they leave the eval set."""
    prefixes: dict[str, str] = {}
    demo_ids: set[str] = set()
    by_rel: dict[str, list[dict]] = {}
    for it in items:
        by_rel.setdefault(it["relation"], []).append(it)
    for rel, its in by_rel.items():
        stable = sorted((x for x in its if x["label"] == "stable"), key=lambda x: -x["sitelinks"])
        demos = stable[:n_shots]
        demo_ids.update(d["id"] for d in demos)
        prefixes[rel] = "".join(f"{d['prompt']} {d['answer']}.\n" for d in demos)
    return prefixes, demo_ids


def full_prompt(prefixes: dict[str, str], item: dict) -> str:
    return prefixes[item["relation"]] + item["prompt"]


def first_token_id(model, text: str) -> int:
    """Token id of the first token of ' <text>' (answers follow a space in our template)."""
    toks = model.to_tokens(" " + text, prepend_bos=False)[0]
    return int(toks[0])


@dataclass
class ModelOutput:
    continuation: str
    resid: np.ndarray  # [n_layers, d_model] residual stream after each block at last prompt token
    top_token: str
    answer_logit: float
    answer_rank: int
    outdated_logit: float | None
    outdated_rank: int | None


@torch.no_grad()
def run_item(
    model, prompt: str, answer_tok: int, outdated_tok: int | None, max_new_tokens: int = 10
) -> ModelOutput:
    logits, cache = model.run_with_cache(
        prompt, names_filter=lambda n: n.endswith("hook_resid_post"), pos_slice=-1
    )
    last = logits[0, -1].float()
    order = torch.argsort(last, descending=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(len(order), device=order.device)
    resid = torch.stack(
        [cache[f"blocks.{i}.hook_resid_post"][0, 0] for i in range(model.cfg.n_layers)]
    )
    text = model.generate(prompt, max_new_tokens=max_new_tokens, do_sample=False, verbose=False)
    continuation = text[len(prompt) :] if text.startswith(prompt) else text
    return ModelOutput(
        continuation=continuation,
        resid=resid.float().cpu().numpy().astype(np.float16),
        top_token=model.to_string([int(order[0])]),
        answer_logit=float(last[answer_tok]),
        answer_rank=int(rank[answer_tok]),
        outdated_logit=float(last[outdated_tok]) if outdated_tok is not None else None,
        outdated_rank=int(rank[outdated_tok]) if outdated_tok is not None else None,
    )
