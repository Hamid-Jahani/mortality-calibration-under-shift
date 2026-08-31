#!/usr/bin/env bash
# Launch the STABLE GP pass on the bastion (.49 / ai-server). Runs ON the
# server; ship + run from the laptop (Gaming profile, single hop):
#
#   cat src/mortcal/runner.py | /c/Windows/System32/OpenSSH/ssh.exe baazar \
#     'cat > ~/mortality-calibration-under-shift/src/mortcal/runner.py'
#   cat scripts/server_launch_stable_gp.sh | /c/Windows/System32/OpenSSH/ssh.exe baazar \
#     'cat > ~/mortality-calibration-under-shift/scripts/server_launch_stable_gp.sh'
#   /c/Windows/System32/OpenSSH/ssh.exe baazar \
#     'bash ~/mortality-calibration-under-shift/scripts/server_launch_stable_gp.sh'
#
# Shipping runner.py first matters: it brings the q995 quantile columns and
# the observed-lifetable fix (2026-08-31), so the stable GP rows never need
# patch_obs_lifetable.py.
#
# Refuses to start while any run_regime.py is alive (pass 1 still finishing)
# unless FORCE=1. Jobs sized by RAM at 3 GB per GP worker, capped at 8; the
# GP 60-year window cap (addendum 4) lives in MODEL_KWARGS, nothing to pass.
# All 20 registered populations: HRV/KOR yield instant design-floor error
# rows, completing the ledger the same way pass 1 did. Parts land in
# results/stable_gp.parts/ (disjoint from pass 1); resume skips done parts.
set -uo pipefail
cd "$(dirname "$0")/.."

if [ "${FORCE:-0}" != "1" ] && pgrep -f '[r]un_regime.py' >/dev/null; then
  echo "REFUSED: run_regime.py still running (pass 1 not finished). FORCE=1 overrides."
  pgrep -af '[r]un_regime.py' | head -3
  exit 1
fi

PY="$PWD/.venv-server/bin/python"
[ -x "$PY" ] || { echo "no interpreter at $PY"; exit 1; }
MEM_GB=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
JOBS="${JOBS:-$(( MEM_GB / 3 < 8 ? MEM_GB / 3 : 8 ))}"
[ "$JOBS" -lt 1 ] && JOBS=1
export MORTCAL_DEVICE=cpu
# thread pinning belongs in the launch environment (measured: load 52 -> 11)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p results/logs

setsid nohup "$PY" scripts/run_regime.py stable --out results/stable_gp.parquet \
  --models GP --jobs "$JOBS" > results/logs/stable_gp_49.out 2>&1 < /dev/null &
disown
echo "launched stable GP: JOBS=$JOBS MEM=${MEM_GB}G log=results/logs/stable_gp_49.out"
sleep "${HEALTH_WAIT:-30}"
echo "--- load: $(cut -d' ' -f1-3 /proc/loadavg) | run_regime procs: $(pgrep -fc '[r]un_regime.py' || true) ---"
tail -3 results/logs/stable_gp_49.out
