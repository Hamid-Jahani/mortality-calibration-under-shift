"""Post-sweep QA gate: run BEFORE scripts/analyse.py on any real-data parquet.

    python scripts/final_qa.py results/shift.parquet [results/shift_gp.parquet ...]

Fails (exit 1) if any row carries a machine-failure error (MemoryError,
partially-initialised torch import, MemoryError-shaped allocation failures):
those are artefacts of a broken run (2026-08-27: a laptop launch under
Windows commit exhaustion), never legitimate cells, and must be re-run, not
analysed around. Structural infeasibility rows ("panel too short",
explosive-draw rejection, admissibility floor) are legitimate: they are
tabulated per (population, model, mechanism) for the paper's design-floor
table and left in place for the common-cell restriction to handle.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

MACHINE = re.compile(r"MemoryError|partially initialized module|Unable to allocate|paging file",
                     re.IGNORECASE)
STRUCTURAL = re.compile(r"panel too short|need >|inadmissible|n_train", re.IGNORECASE)
#: a family's own sampler refusing to produce a predictive law on that panel
#: (addendum 3 §7: rejection only, never clipping). Real-data 2026-08-28:
#: SVAR on TWN 990-1000/1000 draws explosive; bootstrap refits on the longest
#: panels overflow Poisson composition ("lam value too large"). A finding
#: about the family, reported as such - not a design-floor cell.
METHOD = re.compile(r"remain explosive|lam value too large", re.IGNORECASE)


def main(paths: list[str]) -> int:
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        df["_source"] = Path(p).name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    err = df[df["error"].notna()].copy()
    machine = err[err["error"].str.contains(MACHINE)]
    structural = err[err["error"].str.contains(STRUCTURAL) & ~err["error"].str.contains(MACHINE)]
    method = err[err["error"].str.contains(METHOD) & ~err["error"].str.contains(MACHINE)
                 & ~err.index.isin(structural.index)]
    other = err[~err.index.isin(machine.index) & ~err.index.isin(structural.index)
                & ~err.index.isin(method.index)]

    print(f"rows={len(df)}  error_rows={len(err)}  machine={len(machine)}  "
          f"structural={len(structural)}  method={len(method)}  other={len(other)}")
    print(f"devices={sorted(df['device'].dropna().unique().tolist())}  "
          f"regimes={sorted(df['regime'].unique().tolist())}  "
          f"pops={df['pop'].nunique()}  models={df['model'].nunique()}  mechs={df['mechanism'].nunique()}")

    if len(structural):
        print("\nstructural (design-floor) cells — report as such, never as failures:")
        tab = (structural.groupby(["pop", "model", "mechanism"]).size()
               .rename("n").reset_index())
        print(tab.to_string(index=False))
    if len(other):
        print("\nUNCLASSIFIED errors — read every one:")
        print(other[["pop", "sex", "model", "mechanism", "error"]]
              .assign(error=lambda d: d["error"].str[:120]).to_string(index=False))
    if len(machine):
        print("\nMACHINE-FAILURE rows — re-run these (pop, model) parts; do NOT analyse:")
        print(machine.groupby(["_source", "pop", "model"]).size().rename("n").reset_index()
              .to_string(index=False))
        return 1
    print("\nQA PASS: no machine-failure rows; safe to run scripts/analyse.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
