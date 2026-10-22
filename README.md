# Facts with a shelf life: does an LLM know when its answer has expired?

An LLM-reliability evaluation built on mechanistic interpretability. Many facts a language
model memorizes are true only for a while: who runs a company, who governs a country, who
coaches a club. After the training cutoff those facts keep changing, and the model keeps
answering. This project asks whether the model's own activations carry a signal that its answer
is wrong, where in the network the fact is looked up, and whether that signal fails precisely on
the facts that have expired.

Everything here runs on a laptop (Apple Silicon, 32 GB) in an evening, and every label traces
back to specific Wikidata statement IDs in a frozen snapshot.

## Benchmark

Facts are drawn from Wikidata properties whose statements carry start and end dates, so the full
change history of each fact is machine-readable:

| key | property | prompt template |
| --- | --- | --- |
| ceo | P169 chief executive officer | The chief executive officer of {subject} is |
| chair | P488 chairperson | The chairperson of {subject} is |
| head_coach | P286 head coach | The head coach of {subject} is |
| head_of_government | P6 (title from P1313) | The {office} is |
| head_of_state | P35 (title from P1906) | The {office} is |
| director | P1037 director / manager | The director of {subject} is |
| headquarters | P159 headquarters location | The headquarters of {subject} is located in |

Each fact is classed relative to the model's training cutoff C (assumed 2023-12-31 for
Qwen2.5-3B; `--cutoff` overrides) and the snapshot date T:

- **stable**: the current value began at least 5 years before C and still holds at T.
- **superseded**: a different value held at C and the current value began after C. The model
  saw the old value in training; the fact has since expired.
- **recent**: the current value began within 18 months before C and still holds. Present in
  training data, but thinly.

Everything else is excluded with a recorded reason (`data/processed/excluded.jsonl`): undated
current values, several simultaneous current values, dates near the cutoff with only year
precision, changes less than 60 days old, intermediate ages, and subjects or answers without an
English Wikipedia article. The rules live in `src/tfi/labeling.py` and are unit-tested.

Raw SPARQL responses are frozen under `data/raw/` with the query and retrieval time; the
manifest records their SHA-256 hashes. Each statement's Wikidata reference count is recorded so
unsourced statements can be filtered at analysis time.

## Pipeline

```
python -m tfi.build_dataset                     # Wikidata -> data/processed/benchmark.jsonl
python -m tfi.run_model --out results/qwen2.5-3b  # answers, grades, last-token residual stream
python -m tfi.probes   --run results/qwen2.5-3b   # per-layer probes + controls + calibration
python -m tfi.patching --run results/qwen2.5-3b   # activation patching over entity-swap pairs
python -m tfi.figures  --run results/qwen2.5-3b
python -m tfi.export_demo --run results/qwen2.5-3b  # JSON bundle for the website explorer
```

Run from the repo root with `PYTHONPATH=src` (or `pip install -e .`).

**Prompting.** The base model is prompted with three demonstrations per relation (the three most
popular stable facts of that relation, which are then removed from the evaluation set) followed
by the target sentence. Greedy decoding, 10 new tokens; the first clause is graded by alias
match against the current value (`correct`), the value that held at the cutoff (`outdated`), any
older holder (`stale`), or none (`other`).

**Probes.** For each layer, a logistic regression on the residual stream at the last prompt
token predicts whether the answer will be wrong. Splits are entity-disjoint (GroupKFold by
subject); the regularization strength is chosen on inner folds; the layer used for calibration
and per-class analysis is chosen by inner-fold AUROC inside each outer fold. Controls: probes
trained on shuffled labels, 1-D probes on random directions, and the majority class.

**Patching.** For pairs of correctly answered same-relation facts with different first answer
tokens, the corrupted (B) prompt is run while one activation from the clean (A) run is copied in:
residual stream, attention output, or MLP output, at every layer, at the last subject token or
the final token. The metric is the fraction of the A-vs-B logit difference recovered.

## Results

Pending: the first full run is in progress. Numbers and figures will be added here from
`results/qwen2.5-3b/` once it completes.

## Layout

```
src/tfi/      relations, wikidata, labeling, build_dataset, model, run_model,
              probes, patching, figures, export_demo
tests/        labeling and grading unit tests
data/raw/     frozen SPARQL responses (git-ignored, regenerable)
data/processed/  benchmark.jsonl, excluded.jsonl, manifest.json
results/      per-model run artifacts (activations git-ignored)
```

## Requirements

Python 3.12, PyTorch with MPS, TransformerLens 4.x (`TransformerBridge`), transformers,
scikit-learn, pandas, matplotlib. `pip install -e .[dev]`.
