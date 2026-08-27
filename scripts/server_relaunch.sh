#!/usr/bin/env bash
# (Re)launch the sweeps on the compute node, safely from a remote shell.
#
#   bash scripts/server_relaunch.sh shift placebo
#
# Why a script: killing an old launcher with `pkill -f launch_sweeps.sh` from
# an `ssh host "..."` one-liner matches the caller's OWN command line and kills
# it (happened 2026-08-27). The bracket patterns below never match themselves.
set -uo pipefail
cd "$(dirname "$0")/.."
REGIMES="${*:-shift placebo}"

pkill -f '[l]aunch_sweeps.sh' 2>/dev/null
pkill -f '[r]un_regime.py'   2>/dev/null
pkill -f '[m]ultiprocessing.spawn' 2>/dev/null
sleep 3
echo "old run stopped; remaining run_regime procs: $(pgrep -fc '[r]un_regime.py')"

export PATH="$HOME/.local/bin:$PATH"
export MORTCAL_PY="${MORTCAL_PY:-$PWD/.venv-server/bin/python}"
export MORTCAL_DEVICE="${MORTCAL_DEVICE:-cpu}"
export JOBS="${JOBS:-$(( $(nproc) > 2 ? $(nproc) - 2 : 1 ))}"
export GP_JOBS="${GP_JOBS:-$(( $(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo) / 3 ))}"
# thread pinning belongs in the environment (see launch_sweeps.sh header)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p results/logs
setsid nohup bash scripts/launch_sweeps.sh $REGIMES > results/logs/server_launch.out 2>&1 < /dev/null &
disown
echo "launched (JOBS=$JOBS GP_JOBS=$GP_JOBS) regimes: $REGIMES"
sleep "${HEALTH_WAIT:-45}"
echo "--- load: $(cut -d' ' -f1-3 /proc/loadavg) | run_regime procs: $(pgrep -fc '[r]un_regime.py') ---"
for r in $REGIMES; do echo "$r parts: $(ls results/$r.parts 2>/dev/null | wc -l)"; done
grep -E 'population|jobs=|resume|FAILED|Error' results/logs/server_launch.out | tail -4
