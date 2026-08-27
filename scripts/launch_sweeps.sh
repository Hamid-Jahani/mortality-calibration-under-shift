#!/usr/bin/env bash
# Two-pass real-data sweeps for the pre-registered regimes (see .memory,
# "LAUNCH PLAN"). Pass 1: everything except GP, one process per population,
# single-threaded numerics. Pass 2: GP alone at low parallelism (1.6 GB/cell).
# Resumable: existing results/<regime>.parts/<POP>.parquet are skipped.
#
#   bash scripts/launch_sweeps.sh shift placebo        # tonight
#   bash scripts/launch_sweeps.sh stable               # multi-day
#
# MORTCAL_DEVICE=cuda may be exported first ONLY if results/timings_gpu.json
# beat results/timings_cached.json for the neural cells.
set -euo pipefail
cd "$(dirname "$0")/.."
# MORTCAL_PY overrides the interpreter (the venv may live outside the repo:
# UV_PROJECT_ENVIRONMENT=C:/Users/Gaming/venvs/mortcal after the 2026-08-27
# zombie-uv incident left .venv undeletable until a reboot — see .memory).
PY="${MORTCAL_PY:-.venv/Scripts/python.exe}"
JOBS="${JOBS:-12}"
GP_JOBS="${GP_JOBS:-2}"
mkdir -p results/logs

for regime in "$@"; do
  log="results/logs/${regime}_$(date +%Y%m%d_%H%M).log"
  echo "== $regime pass 1 (all but GP, jobs=$JOBS) -> $log"
  "$PY" scripts/run_regime.py "$regime" --out "results/${regime}.parquet" \
        --exclude-models GP --jobs "$JOBS" 2>&1 | tee -a "$log"
  echo "== $regime pass 2 (GP only, jobs=$GP_JOBS)"
  "$PY" scripts/run_regime.py "$regime" --out "results/${regime}_gp.parquet" \
        --models GP --jobs "$GP_JOBS" 2>&1 | tee -a "$log"
done
echo "all requested regimes finished: $*"
