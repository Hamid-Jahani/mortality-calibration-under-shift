"""Project sweep wall time from a per-cell probe.

    uv run python scripts/sweep_cost.py [results/timings.json]

Reads scripts/time_cells.py output — measured on SWE males, the LONGEST
training panel at 269 years, so every number here is an upper bound for
other populations — and multiplies by the admissible grid and the buildable
(origin, population, sex) triples of each regime.

Unprobed cells are imputed from the same FAMILY's cheapest one-fit probe
scaled by the mechanism's fit count (docs/GRID.md: native 1, dropout 1,
split 2, EnbPI/copula 11, ensemble 10, pboot 1+B). Never from another
family: per-fit cost differs by two orders of magnitude across the grid, so
a cross-family imputation is worthless. Imputed entries are marked, and
PRIMARY and SECONDARY cells are totalled separately because the four "(s)"
cells of docs/GRID.md are explicitly run-if-time.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mortcal.runner import ADMISSIBLE, SECONDARY   # noqa: E402

#: buildable (origin, pop, sex) triples per regime. STABLE is the measured
#: audit (520 triples, 119 unbuildable); the other two are populations x
#: sexes after the addendum 3 §4 admissibility floor.
REGIME_TRIPLES = {"shift": 40, "placebo": 22, "stable": 401}

#: base fits per mechanism (docs/GRID.md cost note).
MECH_FITS = {"native": 1, "dropout": 1, "split_conf": 2, "enbpi": 11,
             "copula_conf": 11, "ensemble": 10, "pboot": 201}

#: probes that cost ONE fit plus scoring — the per-family unit of work.
ONE_FIT = ("native", "dropout")


def build_costs(probed: dict) -> tuple[dict, set]:
    """(per-cell seconds, imputed keys) for every admissible cell."""
    unit = {}                                     # family -> one-fit seconds
    for key, v in probed.items():
        fam, mech = key.split("/")
        if mech in ONE_FIT:
            unit[fam] = min(unit.get(fam, float("inf")), v)
    for key, v in probed.items():                 # fall back: divide a multi-fit probe
        fam, mech = key.split("/")
        if fam not in unit and mech in MECH_FITS:
            unit[fam] = v / MECH_FITS[mech]

    per_cell, imputed = {}, set()
    for (m, u) in sorted(ADMISSIBLE):
        key = f"{m}/{u}"
        if key in probed:
            per_cell[(m, u)] = probed[key]
        elif m in unit:
            per_cell[(m, u)] = unit[m] * MECH_FITS.get(u, 1)
            imputed.add((m, u))
        else:
            per_cell[(m, u)] = float("nan")
            imputed.add((m, u))
    return per_cell, imputed


def main(argv=None) -> int:
    path = Path(argv[0]) if argv else ROOT / "results" / "timings.json"
    if not path.exists():
        print(f"no probe at {path}; run scripts/time_cells.py first")
        return 1
    raw = json.loads(path.read_text())
    probed = {k: v for k, v in raw.items() if isinstance(v, (int, float))}
    failed = {k: v for k, v in raw.items() if not isinstance(v, (int, float))}
    per_cell, imputed = build_costs(probed)

    primary = sorted(ADMISSIBLE - SECONDARY)
    print(f"probe: {path.name} — {len(probed)} measured, {len(failed)} failed, "
          f"{len(imputed)} imputed of {len(ADMISSIBLE)} admissible\n")

    def block(cells, label):
        classical = [c for c in cells if c[0] in ("LC", "PLC", "CBD", "RH", "SVAR")]
        neural = [c for c in cells if c not in classical]
        s_all = sum(per_cell[c] for c in cells)
        s_cl = sum(per_cell[c] for c in classical)
        s_nu = sum(per_cell[c] for c in neural)
        print(f"--- {label}: {len(cells)} cells, {s_all/3600:.2f} h per "
              f"(origin, pop, sex)  [classical {s_cl/3600:.3f} h, "
              f"neural {s_nu/3600:.2f} h] ---")
        print(f"{'regime':9s} {'1 core':>11s} {'8 cores':>10s} "
              f"{'classical only, 8 cores':>25s}")
        tot = 0.0
        for regime, n in REGIME_TRIPLES.items():
            t = s_all * n
            tot += t
            print(f"{regime:9s} {t/3600:8.1f} h {t/3600/8:8.1f} h "
                  f"{s_cl*n/3600/8:23.2f} h")
        print(f"{'ALL':9s} {tot/3600:8.1f} h {tot/3600/8:8.1f} h")
        print(f"{'  (shift+placebo only)':22s} "
              f"{s_all*(REGIME_TRIPLES['shift']+REGIME_TRIPLES['placebo'])/3600/8:8.1f} h "
              f"on 8 cores\n")

    block(primary, "PRIMARY cells")
    block(sorted(ADMISSIBLE), "PRIMARY + SECONDARY")

    print("most expensive PRIMARY cells (seconds each):")
    for c in sorted(primary, key=lambda c: -per_cell[c])[:8]:
        tag = "imputed " if c in imputed else "measured"
        print(f"  {c[0]:5s}/{c[1]:12s} {per_cell[c]:8.1f}  ({tag})")
    if failed:
        print("\nFAILED probes — cost unknown, EXCLUDED from every total above:")
        for k, v in failed.items():
            print(f"  {k}: {str(v)[:95]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
