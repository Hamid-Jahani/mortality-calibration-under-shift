"""Recompute observed life-table functionals in runner parquets in place.

    python scripts/patch_obs_lifetable.py results/placebo.parquet results/placebo_gp.parquet

Why this exists (2026-08-31): the runner closed the OBSERVED life table at
the panel's top age even when that age carried no registered deaths, so a
top age with D ~ 0 over positive exposure floored m at 1e-10 and the
open-group closure e_A = 1/m_A exploded (DNK female 1914: observed e_0 =
8.5e6 years). ``mortcal.runner.observed_functionals`` now closes the
observed table at the last age with D >= 0.5 (the addendum 3 §10 zero-death
threshold). Model-side samples, scores and intervals are untouched by both
the defect and this patch: only ``{e0,e65,ann65}_obs`` and the derived
``*_error`` columns are recomputed, from the manifest-pinned panel, through
the SAME code path the fixed runner uses — so a re-run sweep and a patched
parquet are identical on these columns.

Every changed value is reported. The parquet is rewritten only when
something changed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mortcal.data.hmd import build_panel          # noqa: E402
from mortcal.runner import observed_functionals   # noqa: E402

DEATHS = ROOT / "Dataset" / "deaths" / "Deaths_1x1" / "Deaths_1x1.txt"
EXPOS = ROOT / "Dataset" / "exposures" / "Exposures_1x1" / "Exposures_1x1.txt"
KEYS = ("e0", "e65", "ann65")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parquet", nargs="+", help="runner parquet(s), patched in place")
    ap.add_argument("--age-max", type=int, default=99)
    args = ap.parse_args(argv)

    dfs = {p: pd.read_parquet(p) for p in args.parquet}
    pops = sorted({str(v) for df in dfs.values() for v in df["pop"].unique()})
    print(f"[patch_obs] panel for {len(pops)} populations ...", flush=True)
    panel = build_panel(DEATHS, EXPOS, pops=pops, age_max=args.age_max)

    cache: dict[tuple, dict | None] = {}

    def obs_for(pop: str, sex: str, year: int, age_hi: int):
        """Replicate the runner exactly: the observed table is the first test
        year's D/E on ages 0..derived_age_hi (the row's own recorded block,
        addendum 3 §3), fed through the same ``observed_functionals``."""
        k = (pop, sex, year, age_hi)
        if k not in cache:
            sub = (panel[(panel["pop"] == pop) & (panel["sex"] == sex)
                         & (panel["year"] == year)].sort_values("age"))
            ages = sub["age"].to_numpy()
            if (len(sub) == 0 or ages[0] != 0
                    or not np.array_equal(ages, np.arange(len(ages)))
                    or age_hi >= len(sub)):
                # the runner's pivot would have recorded an error row here;
                # never invent an observed value the sweep could not have had
                cache[k] = None
            else:
                cache[k] = observed_functionals(
                    sub["D"].to_numpy(float)[:age_hi + 1],
                    sub["E"].to_numpy(float)[:age_hi + 1])
        return cache[k]

    for path, df in dfs.items():
        if f"{KEYS[0]}_obs" not in df.columns:
            print(f"[patch_obs] {path}: no derived-quantity columns, skipped")
            continue
        ok_idx = df.index[df["error"].isna()] if "error" in df.columns else df.index
        changes: dict[tuple, int] = {}
        for i in ok_idx:
            row = df.loc[i]
            hi = row.get("derived_age_hi", np.nan)
            if not np.isfinite(hi):
                continue
            vals = obs_for(str(row["pop"]), str(row["sex"]),
                           int(row["origin"]) + 1, int(hi))
            if vals is None:
                continue
            for key in KEYS:
                new_obs = vals[key]
                old_obs = float(row[f"{key}_obs"])
                if bool(np.isclose(new_obs, old_obs, rtol=1e-9, atol=1e-12,
                                   equal_nan=True)):
                    continue
                df.at[i, f"{key}_obs"] = new_obs
                point = float(row[f"{key}_point"])
                df.at[i, f"{key}_error"] = (point - new_obs
                                            if np.isfinite(point) and np.isfinite(new_obs)
                                            else float("nan"))
                ck = (str(row["pop"]), str(row["sex"]), int(row["origin"]), key,
                      round(old_obs, 6) if np.isfinite(old_obs) else float("nan"),
                      round(new_obs, 6) if np.isfinite(new_obs) else float("nan"))
                changes[ck] = changes.get(ck, 0) + 1
        if not changes:
            print(f"[patch_obs] {path}: nothing to change")
            continue
        for (pop, sex, origin, key, old, new), n in sorted(changes.items()):
            print(f"[patch_obs] {path}: {pop}/{sex}/origin={origin} {key}_obs "
                  f"{old} -> {new}  ({n} rows)")
        df.to_parquet(path, index=False)
        print(f"[patch_obs] {path}: rewritten "
              f"({sum(changes.values())} value changes across "
              f"{len({c[:3] for c in changes})} (pop, sex, origin) units)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
