#!/usr/bin/env bash
# Pull sweep results back from the compute node through the two-hop route
# (laptop -> bastion `baazar` -> 192.168.1.47), the mirror of the upload in
# docs/SERVER.md. Streams a tar of results/*.parquet, the parts directories
# and the logs into the local results/. Existing local files are overwritten
# by the node's copy (the node is the source of truth while a sweep runs).
#
#   bash scripts/server_pull.sh            # everything under results/
#   bash scripts/server_pull.sh shift      # one regime (final + parts + log)
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
NODE_REPO='~/mortality-calibration-under-shift'
if [ $# -eq 0 ]; then
  PATHS='results/*.parquet results/*.parts results/logs'
else
  PATHS=""; for r in "$@"; do PATHS="$PATHS results/${r}.parquet results/${r}_gp.parquet results/${r}.parts results/${r}_gp.parts results/logs/${r}_*.log"; done
fi
"$SSH" -o BatchMode=yes baazar "ssh 192.168.1.47 \"cd $NODE_REPO && tar czf - \$(ls -d $PATHS 2>/dev/null | tr '\n' ' ')\"" | tar xzf - -C .
echo "pulled into results/:"; ls -la results/*.parquet 2>/dev/null | awk '{print "  " $5 " " $9}'
for d in results/*.parts; do [ -d "$d" ] && echo "  $(ls "$d" | wc -l) parts in $d"; done
