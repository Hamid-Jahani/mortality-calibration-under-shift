#!/usr/bin/env bash
# Pull the STABLE sweep from BOTH machines and assemble results/stable.parquet.
#
# The stable regime is split across two machines (docs/STATUS.md 2026-08-29):
#   192.168.1.47 (two hops)  — 14 populations, parts named POP__MODEL.parquet
#   baazar / .49 (one hop)   — SWE,DNK,ISL,BEL,NOR,CHE with --origins halves,
#                              parts named POP__MODEL__o<first>-<last>.parquet
# server_pull.sh assumes everything lives on .47, so stable needs this
# two-source variant. Part names are disjoint by construction (different
# populations AND the origin tag), and the assembly aborts on any duplicate
# (regime, pop, sex, origin, model, mechanism) cell rather than averaging.
#
#   bash scripts/server_pull_stable.sh              # into results/ (the real pull)
#   bash scripts/server_pull_stable.sh <dir>        # dry-run into another dir
#
# The machines' final stable.parquet files are deliberately NOT pulled: each
# is a concat of that machine's parts only, misleading locally. Assembly
# mirrors scripts/run_regime.py (pd.concat of all parts, ignore_index).
#
# AFTER the real pull, before QA:
#   python scripts/patch_obs_lifetable.py results/stable.parquet [results/stable_gp.parquet]
# — the server runners predate the observed-lifetable fix (2026-08-31).
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="${1:-results}"
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REPO='~/mortality-calibration-under-shift'
PY=C:/Users/Gaming/venvs/mortcal-cpu/Scripts/python.exe

mkdir -p "$DEST"
# the machines are the source of truth: purge local snapshots first so stale
# laptop parts can never be merged into a pull (same rule as server_pull.sh)
rm -rf "$DEST/stable.parts" "$DEST/stable_gp.parts"

echo "== 192.168.1.47 (two-hop): untagged parts =="
"$SSH" -o BatchMode=yes baazar \
  "ssh 192.168.1.47 \"cd $REPO && tar czf - --ignore-failed-read results/stable.parts results/stable_gp.parts results/logs/stable* 2>/dev/null\"" \
  | tar xzf - -C "$DEST" --strip-components=1

echo "== baazar / .49 (single-hop): origin-tagged parts =="
"$SSH" -o BatchMode=yes baazar \
  "cd $REPO && tar czf - --ignore-failed-read results/stable.parts results/stable_gp.parts results/logs/stable* 2>/dev/null" \
  | tar xzf - -C "$DEST" --strip-components=1

N47=$(ls "$DEST/stable.parts" 2>/dev/null | grep -vc '__o' || true)
N49=$(ls "$DEST/stable.parts" 2>/dev/null | grep -c '__o' || true)
echo "parts: .47=$N47 (expect 126 when complete)  .49=$N49 (expect 108 when complete)"

"$PY" - "$DEST" <<'PYEOF'
import sys
from pathlib import Path

import pandas as pd

dest = Path(sys.argv[1])
parts = sorted((dest / "stable.parts").glob("*__*.parquet"))
if not parts:
    raise SystemExit("no parts pulled")
df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
# same assembly as scripts/run_regime.py; a duplicate cell would mean the two
# machines overlapped a population -- abort loudly, never average
key = ["regime", "pop", "sex", "origin", "model", "mechanism"]
dup = df.duplicated(key, keep=False)
if dup.any():
    raise SystemExit(f"{int(dup.sum())} duplicate cells across machines, e.g. "
                     f"{df.loc[dup, key].iloc[0].to_dict()}; fix upstream")
out = dest / "stable.parquet"
df.to_parquet(out, index=False)
err = int(df["error"].notna().sum()) if "error" in df else 0
pops = df["pop"].nunique()
print(f"assembled {out}: rows={len(df)} error_rows={err} parts={len(parts)} pops={pops}")
PYEOF

echo "done. next: python scripts/patch_obs_lifetable.py $DEST/stable.parquet"
