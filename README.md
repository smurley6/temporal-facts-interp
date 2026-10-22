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

## Results (Qwen2.5-3B, benchmark snapshot 2026-10-22, cutoff 2023-12-31)

All numbers below are produced by the commands above and live in `results/qwen2.5-3b/`.
Three demonstration facts per relation are held out, leaving 2,742 evaluated facts. The full
run took 2 h 23 m for the model pass and about 3 h 20 m for 120 patching pairs on an M5 laptop.

### What the model knows

A 3B base model knows few of these facts: 289 of 2,742 (10.5%) are answered correctly. The
rest split into answers that name a real former holder and answers that name nobody relevant:

| grade | n | meaning |
| --- | --- | --- |
| correct | 289 | current value |
| outdated | 103 | the value that held at the training cutoff (superseded facts only) |
| stale | 292 | an even older holder |
| other | 2,058 | no past or present holder named |

Knowledge is concentrated in heads of state (60% name a real holder), CEOs (45%) and
headquarters (50%); head coaches and institutional directors are mostly unknown (17% and 13%).

On **superseded** facts the model is correct 4 times out of 913. Where it names anyone real
(238 facts), it gives the value that held at the cutoff 43% of the time and an even older one
55% of the time: Bob Iger for Disney, Carlos Tavares for Stellantis, Hans Vestberg for Verizon.
Facts whose current value took office in 2022 or 2023 are answered correctly 1% of the time,
against 15 to 25% for 2015 to 2019 (`fig_cutoff.png`), so the model's effective knowledge of
this kind of fact ends well before the nominal cutoff. See *Caveats*.

### Probes: does the residual stream predict an error?

**All facts** (`probes/`, n = 2,742, error rate 89%). A linear probe on the residual stream at
the last prompt token reaches AUROC 0.87 at layer 30 of 36 on entity-disjoint folds; shuffled
labels give 0.50 and random directions 0.61. Plain accuracy (0.90) is uninformative because the
majority class is 0.89. The probe transfers to relations held out entirely (AUROC 0.70 to 0.86
for six of seven; headquarters 0.60). Calibration is good (ECE 0.02).

**Known facts only** (`probes_known/`, the 684 facts where the model named a real past or
present holder, 42% correct). This is the sharper question: given that the model knows the
entity, does its residual stream know the knowledge is out of date? AUROC 0.77 at layer 13,
accuracy 0.71 against a 0.58 majority; shuffled labels 0.51, random direction 0.54.

The breakdown by grade at the selected layer:

| grade | n | mean probe P(error) |
| --- | --- | --- |
| correct | 289 | 0.44 |
| outdated | 103 | 0.67 |
| stale | 292 | 0.69 |

and by shelf-life class: stable 0.44, recent 0.68, superseded 0.73.

**What the signal is.** Within the stable class alone (280 correct vs 41 stale), the probe is
at chance (AUROC 0.51). Within the superseded class there are only 4 correct answers to
separate from 234 wrong ones, and the probe does not separate them. So the "error signal" the
probe reads is really a *volatility* signal: the residual stream encodes that this is the kind
of fact that changes (a coach, a prime minister), and that correlates with being wrong because
such facts have usually changed. It does not encode "my specific answer is out of date". The
misses follow directly: at the base-rate threshold the probe still lets 21% of superseded errors
and 28% of recent errors through, and its false-alarm rate on superseded facts is 100% in the
trivial sense that it flags everything in that class.

### Activation patching: where is the fact looked up?

120 entity-swap pairs (same relation, both answered correctly, different first answer tokens).
`fig_patching.png`:

- Patching the **residual stream at the last subject token** recovers up to 75% of the logit
  difference, sustained from layer 5 to layer 27 and gone by layer 31.
- Patching the **residual stream at the final token** recovers nothing until layer 27 and 100%
  from layer 31: the answer is assembled at the final position in the last third of the network.
- **Attention output at the final token** carries the transfer (layers 27 and 31).
- Single-layer **MLP output at the subject token** recovers at most 17%; a 5-layer window of
  MLPs recovers 41 to 48% at layers 6 to 11 and under 10% beyond layer 15. The lookup is spread
  over early-to-mid MLPs. The middle third of the network (layers 12 to 23) accounts for 16% of
  windowed MLP recovery, not the majority.

### Cutoff sensitivity: the same pipeline with cutoff 2021-12-31

The accuracy-by-year curve says the model's knowledge of appointments ends well before
2023-12-31, so the whole pipeline was rerun with `--cutoff 2021-12-31` (benchmark in
`data/processed_cutoff2021/`, run in `results/qwen2.5-3b-cutoff2021/`, same Wikidata snapshot
date, 60 patching pairs instead of 120). Moving the cutoff relabels the benchmark: 393 facts whose
holder started in 2017 or 2018 drop out of *stable* into the excluded intermediate-age band, 346
facts whose holder started between mid-2020 and the end of 2021 enter *recent*, and 275 of the
old *recent* facts (holders who took office in 2022 or 2023) become *superseded*, since a
predecessor now held at the cutoff. 1,927 facts are in both benchmarks.

| | cutoff 2023-12-31 | cutoff 2021-12-31 |
| --- | --- | --- |
| evaluated facts (stable / recent / superseded) | 2,742 (1,283 / 546 / 913) | 2,486 (891 / 346 / 1,249) |
| model correct: stable | 21.8% | 22.0% |
| model correct: recent | 0.9% | 4.0% |
| model correct: superseded | 0.4% | 0.6% |
| superseded facts where a real holder is named: cutoff-era value vs older | 43% vs 55% (n = 238) | 57% vs 41% (n = 362) |
| probe AUROC, all facts (peak layer) | 0.87 (layer 30) | 0.86 (layer 30) |
| probe AUROC, known facts (selected layer) | 0.77 (layer 13, n = 684) | 0.78 (layer 17, n = 649) |
| within-stable AUROC, known facts | 0.51 (280 correct vs 41 stale) | 0.50 (196 vs 17) |
| patching: residual at subject token, peak and span above 50% | 75% at layer 15, layers 0 to 26 | 71% at layer 14, layers 1 to 25 |
| patching: residual at final token, onset and full recovery | 53% at layer 27, 100% from 32 | 51% at layer 27, 100% from 32 |
| patching: 5-layer MLP window at subject token, peak | 48% at layer 8 | 42% at layer 8 |

What the rerun changes:

- **No cutoff makes the recent class known.** With the cutoff at the end of 2021, facts whose
  holder took office in the preceding 18 months are answered correctly 4% of the time (14 of
  346), against 22% for stable facts. Per start year (`fig_cutoff.png` in each run), accuracy
  is 20 to 25% for holders who started in 2015 to 2017, 14 to 16% for 2018 and 2019, 4 to 5%
  for 2020 and 2021, and about 1% from 2022 on. Knowledge of this fact type fades through the
  late 2010s rather than stopping at a date, so the three-class design overstates how sharply
  "in training" and "after training" separate for a 3B model.
- **The earlier cutoff describes the model's answers better.** When the model names a real
  holder for a superseded fact, it now gives the value that held at the cutoff 57% of the time
  (43% with the 2023 cutoff), and an even older one 41% of the time. Overall accuracy falls
  from 10.5% to 8.8% because the 2021 benchmark has more superseded facts. The demonstration
  facts are re-chosen from the new stable set, so the model's continuations differ on many of
  the 1,906 shared facts, but the grade agrees on 90% of them and the correct-to-wrong flips
  are balanced (21 lost, 15 gained); the 92 stale-to-outdated changes are relabeling, since the
  predecessor at the new cutoff is a different person.
- **Probe conclusions are unchanged.** AUROC on all facts and on known facts moves by about
  0.01 in each direction, controls stay at chance, and the within-stable AUROC is again 0.50:
  the probe reads volatility, not the correctness of the specific answer, whichever cutoff is
  assumed. The known-facts probe now selects layer 17 in every fold (13 before, with one fold
  at 30), and the all-facts probe's per-class AUROC shifts with the class membership (recent
  0.62 to 0.70, superseded 0.67 to 0.55), as expected when 275 facts move between those classes.
- **Patching is the same circuit.** With half the pairs, every peak lands within one layer of
  the first run and the recoveries are within 6 points.

### Caveats

- **Cutoff.** 2023-12-31 is assumed, not measured. The rerun with cutoff 2021-12-31 above
  shows the conclusions do not depend on the choice, but also that the model's knowledge of
  appointments fades through 2017 to 2021 rather than ending at any one date, so "recent" is a
  thinly known class under either cutoff.
- **Scale.** At 3B, 89% of facts are unknown, so class-level statistics lean on a few hundred
  known facts. The pipeline is model-agnostic; an 8B run changes only `--model` and the GPU.
- **Wikidata is the ground truth.** Labels inherit its errors. Every item records its statement
  IDs and reference counts so unsourced statements can be filtered.
- **Within-class AUROC** is reported but near-meaningless for superseded and recent facts,
  where the model is almost never correct.

## Layout

```
src/tfi/      relations, wikidata, labeling, build_dataset, model, run_model,
              probes, patching, figures, export_demo
tests/        labeling and grading unit tests
data/raw/     frozen SPARQL responses (git-ignored, regenerable)
data/processed/  benchmark.jsonl, excluded.jsonl, manifest.json (cutoff 2023-12-31)
data/processed_cutoff2021/  the same, built with --cutoff 2021-12-31
results/      per-model run artifacts (activations git-ignored)
```

## Requirements

Python 3.12, PyTorch with MPS, TransformerLens 4.x (`TransformerBridge`), transformers,
scikit-learn, pandas, matplotlib. `pip install -e .[dev]`.
