"""Project total sweep wall time from results/timings.json.

    uv run python scripts/sweep_cost.py

Reads the per-cell probe (scripts/time_cells.py, measured on SWE males —
the LONGEST training panel at 269 years, so these are upper bounds for every
other population) and multiplies by the admissible grid and the buildable
(origin, population, sex) triples of each regime.

Unprobed cells are imputed from their family's cheapest probed mechanism
scaled by that mechanism's cost ratio elsewhere; imputed entries are marked
so no number here is mistaken for a measurement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mortcal.runner import ADMISSIBLE, SECONDARY   # noqa: E402

#: buildable (origin, pop, sex) triples per regime — the STABLE figure is the
#: measured audit (520 triples, 119 unbuildable), the other two are
#: populations x sexes with the addendum 3 §4 admissibility floor applied.
REGIME_TRIPLES = {"shift": 40, "placebo": 22, "stable": 401}

#: mechanism cost multipliers relative to that family's native cell, from the
#: fit counts documented in docs/GRID.md (native 1, pboot 1+B, split 2,
#: enbpi/copula K+1, ensemble M, dropout 1).
MECH_FITS = {"native": 1, "pboot": 201, "split_conf": 2, "enbpi": 11,
             "copula_conf": 11, "ensemble": 10, "dropout": 1}


def main() -> int:
    path = ROOT / "results" / "timings.json"
    if not path.exists():
        print("run scripts/time_cells.py first"); return 1
    raw = json.loads(path.read_text())
    probed = {k: v for k, v in raw.items() if isinstance(v, (int, float))}
    failed = {k: v for k, v in raw.items() if not isinstance(v, (int, float))}

    fam_native = {k.split("/")[0]: v for k, v in probed.items()
                  if k.endswith("/native")}
    print(f"probed {len(probed)} cells, {len(failed)} failed\n")

    per_cell, imputed = {}, set()
    for (m, u) in sorted(ADMISSIBLE):
        key = f"{m}/{u}"
        if key in probed:
            per_cell[(m, u)] = probed[key]
            continue
        base = fam_native.get(m)
        if base is None:
            base = max(fam_native.values()) if fam_native else 0.0
            imputed.add((m, u))
        per_cell[(m, u)] = base * MECH_FITS.get(u, 1)
        imputed.add((m, u))

    print(f"{'regime':9s} {'cells':>7s} {'1 core':>12s} {'8 cores':>10s}")
    grand = 0.0
    for regime, triples in REGIME_TRIPLES.items():
        total = sum(per_cell[(m, u)] for (m, u) in ADMISSIBLE) * triples
        grand += total
        n_cells = len(ADMISSIBLE) * triples
        print(f"{regime:9s} {n_cells:7d} {total/3600:9.1f} h {total/3600/8:8.1f} h")
    print(f"{'ALL':9s} {'':7s} {grand/3600:9.1f} h {grand/3600/8:8.1f} h")

    print("\nmost expensive cells (per cell, seconds):")
    for (m, u), s in sorted(per_cell.items(), key=lambda kv: -kv[1])[:10]:
        tag = " (imputed)" if (m, u) in imputed else " (measured)"
        star = " [secondary]" if (m, u) in SECONDARY else ""
        print(f"  {m:5s}/{u:12s} {s:9.1f}{tag}{star}")

    classical = {(m, u) for (m, u) in ADMISSIBLE
                 if m in ("LC", "PLC", "CBD", "RH", "SVAR")}
    c_tot = sum(per_cell[c] for c in classical) * sum(REGIME_TRIPLES.values())
    print(f"\nclassical-only (5x5, all regimes): {c_tot/3600:.1f} h "
          f"1-core / {c_tot/3600/8:.1f} h 8-core")
    if failed:
        print("\nFAILED probes (cost unknown, excluded above):")
        for k, v in failed.items():
            print(f"  {k}: {str(v)[:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
