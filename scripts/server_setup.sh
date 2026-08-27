#!/usr/bin/env bash
# One-shot setup + launch of the pre-registered sweeps on a Linux CPU server.
#
#   git clone git@github.com:sheperd007/mortality-calibration-under-shift.git
#   cd mortality-calibration-under-shift
#   # copy the HMD bulk files into ./Dataset/ (see docs/SERVER.md — the data
#   # are registration-restricted, so they travel with you, not with the repo)
#   bash scripts/server_setup.sh            # installs env, verifies, launches
#
# Env knobs: JOBS (default: cores-2), GP_JOBS (default: RAM_GB/3, min 1),
# REGIMES (default "shift placebo"), SKIP_LAUNCH=1 to only prepare.
set -euo pipefail
cd "$(dirname "$0")/.."

# ---- 1. uv + environment (CPU torch: the PyPI Linux wheel is CUDA-bundled,
#         ~2.5 GB and useless here, so torch comes from the cpu index) -------
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$PWD/.venv-server}"
uv python install 3.12 >/dev/null
uv sync                                  # default deps only (no neural group)
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
uv pip install --python "$PY" "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$PY" "gpytorch==1.15.2"   # torch already satisfied -> no CUDA pull
"$PY" - <<'EOF'
import torch, gpytorch, mortcal
print("env OK: torch", torch.__version__, "gpytorch", gpytorch.__version__)
EOF

# ---- 2. data + gates ----------------------------------------------------
test -s Dataset/deaths/Deaths_1x1/Deaths_1x1.txt || { echo "Dataset/ missing — see docs/SERVER.md"; exit 2; }
sha256sum -c --ignore-missing --quiet data/MANIFEST.sha256 && echo "data vintage verified (2026-06-15)"
"$PY" -m pytest tests/ -q -x --no-header -p no:cacheprovider 2>&1 | tail -1

# ---- 3. launch (detached, resumable, logs in results/logs) --------------
CORES=$(nproc); RAM_GB=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
export MORTCAL_PY="$PY" MORTCAL_DEVICE=cpu
export JOBS="${JOBS:-$(( CORES > 2 ? CORES - 2 : 1 ))}"
export GP_JOBS="${GP_JOBS:-$(( RAM_GB / 3 > 0 ? RAM_GB / 3 : 1 ))}"
echo "cores=$CORES ram=${RAM_GB}GB -> JOBS=$JOBS GP_JOBS=$GP_JOBS"
if [ "${SKIP_LAUNCH:-0}" = "1" ]; then echo "prepared; not launching"; exit 0; fi
mkdir -p results/logs
nohup bash scripts/launch_sweeps.sh ${REGIMES:-shift placebo} \
  > results/logs/server_launch.out 2>&1 &
echo "launched pid $! — follow with: tail -f results/logs/server_launch.out"
