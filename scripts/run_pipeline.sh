#!/usr/bin/env bash
# Full analysis pipeline after the benchmark exists. Usage: scripts/run_pipeline.sh [run_dir] [n_pairs]
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${1:-results/qwen2.5-3b}"
PAIRS="${2:-120}"
export PYTHONPATH=src PYTHONUNBUFFERED=1
PY=.venv/bin/python
echo "== run_model  $(date)"; $PY -m tfi.run_model --out "$RUN"
echo "== probes     $(date)"; $PY -m tfi.probes --run "$RUN"
echo "== patching   $(date)"; $PY -m tfi.patching --run "$RUN" --pairs "$PAIRS"
echo "== figures    $(date)"; $PY -m tfi.figures --run "$RUN"
echo "== export     $(date)"; $PY -m tfi.export_demo --run "$RUN"
echo "== done       $(date)"
