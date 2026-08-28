"""Pre-registered sensitivity slices -> results/sensitivities.json.

    python scripts/sensitivities.py --out results/sensitivities.json
    python scripts/sensitivities.py --shift results/_shift_snapshot.parquet \\
                                    --out results/sensitivities.json

Consumes runner rows (``results/<regime>.parquet`` plus
``results/<regime>_gp.parquet`` once the GP pass has landed); refits nothing
(docs/PIPELINE.md). A regime whose parquet is absent is written as the literal
string ``"pending"`` under every top-level key that has a slot for it — never
omitted, never a crash.

Every number is a mean over VALID rows only. A row is valid when ``error`` is
null and the target column(s) are finite. Error rows never enter a
denominator: they are counted per leaf (``n_error_rows``) and classified per
slice (``_meta.error_classes``: machine / design_floor / method / other — the
regexes of ``scripts/final_qa.py``, imported so there is one definition).

What is computed, and where it is registered
--------------------------------------------
* PREREGISTRATION-ADDENDUM-1 §A — placebo strata (pooled, neutral,
  belligerent_total, civilian_only) with per-stratum coverage_95 and
  joint_path_coverage_95 per family x mechanism; sensitivities drop_GBR_SCO,
  neutral_only, DNK with / without test years 1921-22 (horizons 8-9).
* PREREGISTRATION-ADDENDUM-1 §B — shift sensitivities: drop_2024 (S-S1 =
  horizon 5), drop_USA_CHL (S-S2), register_based vs census_based (S-S3).
  The two population sets follow docs/DATA-PREREQS.md §B.2 item 9 and are
  stated under ``definitions.shift_population_sets``.
* PREREGISTRATION-ADDENDUM-3 §4 — per-origin effective cluster count
  (populations with valid rows) per regime, per arm and panel-wide.
* PREREGISTRATION-ADDENDUM-3 §11 — per family x mechanism, the number of
  cells lost to the common-cell restriction relative to the full panel, for
  each contrast sub-grid: all primary arms, within one family (the mechanism
  contrast), within one mechanism (the family contrast).

Horizon-subset slices (``drop_2024``, ``dnk_drop_1921_1922``) recompute
coverage_95 as the mean of the retained ``coverage95_h*`` columns; with every
horizon retained this reproduces the runner's coverage_95 exactly (the age
mask is common to all horizons; verified to 2e-16 on real rows). Joint path
coverage over a horizon SUBSET is not recoverable from runner rows — only the
full-path indicator is emitted — so those leaves carry
``joint_path_coverage_95: null`` and ``joint_path_coverage_95_lower_bound``
= the full-path value (dropping a horizon can only turn a path miss into a
hit, never the reverse).

Scoring discipline (addendum 2 §3; runner docstring)
----------------------------------------------------
Only coverage_95 and joint_path_coverage_95 are aggregated. Both are valid
for every mechanism, including the conformal arms, whose crps / logscore /
PIT are placeholders (``scores_secondary``) and whose coverage_50/80 are NaN
by design. No proper score appears anywhere in this output. Arms are never
averaged together: each leaf is exactly one family/mechanism and carries its
own age support (``n_ages_scored``; CBD = 45 ages, 55-99), so make_tables.py
keeps CBD in its own row block with the support stated.

JSON CONTRACT (consumed by scripts/make_tables.py)
--------------------------------------------------
Top level::

    {
      "contract_version": 1,
      "generated": "<ISO-8601 UTC timestamp>",
      "snapshot": bool,      # True if ANY source basename starts with "_"
                             # (a partial snapshot). Every table built from
                             # such a file MUST carry the first-line comment
                             # "% GENERATED SNAPSHOT - NOT FINAL - regenerate
                             #  from results/<regime>.parquet".
      "sources": {regime: [paths read] | null},
      "definitions": {...},  # registered population sets, horizon sets,
                             # notes; see ``definitions()``
      "strata":             {"placebo": <regime block>},
      "sensitivities":      {"shift": <regime block>, "placebo": <regime block>},
      "effective_clusters": {"shift": <rb>, "placebo": <rb>, "stable": <rb>},
      "common_cell_losses": {"shift": <rb>, "placebo": <rb>, "stable": <rb>}
    }

    <regime block> = "pending"                 # parquet absent
                   | {slice_name: <slice>, ...}
    <slice>  = {"_meta": {...}, "FAMILY/MECHANISM": <leaf>, ...}
               Keys beginning with "_" are metadata, never arms.
    <leaf>   = {
        "coverage_95":            float | null,
        "joint_path_coverage_95": float | null,
        "n_cells":        int,        # valid rows (= cells) for this arm in the slice
        "n_error_rows":   int,        # error rows for this arm in the slice
        "n_ages_scored":  int | null, # modal age support of the valid rows
        "conformal":      bool,       # interval arm: 95% level only, no proper score
        "grid_secondary": bool,       # "(s)" cell of docs/GRID.md
        ...                           # slice-specific extras, below
    }
    A cell is one (pop, sex, origin) triple; one runner row per cell and arm.

Slice names and extras::

    strata/placebo         pooled | neutral | belligerent_total | civilian_only
    sensitivities/placebo  full_panel | drop_GBR_SCO | neutral_only |
                           dnk_full_window | dnk_drop_1921_1922
    sensitivities/shift    full_panel | drop_2024 | drop_USA_CHL |
                           register_based | census_based
        horizon-subset slices add: "horizons": [int],
                                   "joint_path_coverage_95_lower_bound": float|null
    effective_clusters/*   origin_<T>, one per origin in the file
        leaf extras: "n_clusters" (populations with >= 1 valid row for the
                     arm), "populations" (sorted codes)
        _meta: n_populations_registered, n_populations_with_rows,
               n_clusters_any_valid_row, n_clusters_all_primary_arms,
               n_design_floor_rows (addendum 3 §4 floor), ...
    common_cell_losses/*   full_panel | all_primary_arms |
                           within_family:<FAM> | within_mechanism:<MECH>
        leaf extras: "n_cells_full" (valid cells, uncensored),
                     "n_cells_lost" (= n_cells_full - n_cells),
                     "populations_lost" (sorted codes),
                     "age_support_block" (within_mechanism only: the family's
                     modal n_ages_scored; intersections are taken per block so
                     CBD never censors the full-age families),
                     "age_support_mismatch": true (within_mechanism only, present
                     only when the arm's own modal support differs from its block)
        _meta: n_cells_common, intersecting_arms, n_cells_panel, primary_arms,
               arms_with_no_valid_rows; for within_mechanism:* the first two
               are dicts keyed by age-support block ("100", "45", ...) and
               _meta.age_support_blocks lists the families per block.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from final_qa import MACHINE, METHOD, STRUCTURAL              # noqa: E402
from mortcal.runner import CONFORMAL_MECHANISMS, SECONDARY    # noqa: E402
from mortcal.splits import (PLACEBO, PLACEBO_POPS, SHIFT,      # noqa: E402
                            SHIFT_POPS)

CONTRACT_VERSION = 1
PENDING = "pending"
META = "_meta"
CELL_KEYS: tuple[str, ...] = ("pop", "sex", "origin")
REGIMES: tuple[str, ...] = ("shift", "placebo", "stable")

# ---------------------------------------------------------------------------
# Registered population sets. Transcribed, not derived — changing one is a
# protocol deviation, not a refactor.
# ---------------------------------------------------------------------------

#: PREREGISTRATION-ADDENDUM-1 §A, strata table.
PLACEBO_STRATA: dict[str, tuple[str, ...]] = {
    "neutral": ("CHE", "DNK", "FIN", "ISL", "NLD", "NOR", "SWE"),
    "belligerent_total": ("FRATNP", "GBRTENW", "ITA"),
    "civilian_only": ("GBR_SCO",),
}

#: docs/DATA-PREREQS.md §B.2 item 9 (S-S3): register-based subset. The
#: complement of SHIFT_POPS is the census-based subset — every population
#: whose 2021-24 exposures are post-censal and therefore provisional under
#: HMD Methods Protocol v6 §5.2.3.
REGISTER_BASED: tuple[str, ...] = ("CHE", "DNK", "FIN", "ISL", "NOR", "SWE")
CENSUS_BASED: tuple[str, ...] = tuple(p for p in SHIFT_POPS if p not in REGISTER_BASED)

#: Addendum-1 §B sensitivity 1 (S-S2).
SHIFT_DROP_POPS: tuple[str, ...] = ("USA", "CHL")

REGISTER_CENSUS_NOTE = (
    "Sets follow docs/DATA-PREREQS.md §B.2 item 9 (S-S3): register-based = "
    "population-register Nordics + CHE (no census cycle, exposures final); "
    "census-based = every other shift population (post-censal 2021-24 "
    "exposures, provisional per HMD Methods Protocol v6 §5.2.3). "
    "PREREGISTRATION-ADDENDUM-1 §B's prose lists EST/LVA/LTU as "
    "register-based in passing; DATA-PREREQS §B.2 places them census-based "
    "(2021-census post-censal inputs). The DATA-PREREQS definition is used "
    "here; the discrepancy is recorded, not silently resolved."
)


def _check_registry() -> None:
    strata = [p for pops in PLACEBO_STRATA.values() for p in pops]
    if sorted(strata) != sorted(PLACEBO_POPS):
        raise RuntimeError("placebo strata do not partition PLACEBO_POPS")
    if sorted(REGISTER_BASED + CENSUS_BASED) != sorted(SHIFT_POPS):
        raise RuntimeError("register/census sets do not partition SHIFT_POPS")
    if not set(SHIFT_DROP_POPS) <= set(SHIFT_POPS):
        raise RuntimeError("SHIFT_DROP_POPS outside SHIFT_POPS")


_check_registry()


def horizons_excluding(regime, years: Iterable[int]) -> list[int]:
    """Registered horizons of `regime` whose test year is NOT in `years`."""
    drop = set(years)
    return [h for h, y in zip(regime.horizons, regime.test_years) if y not in drop]


#: Horizon sets of the two horizon-subset slices, from the Regime objects.
SHIFT_DROP_2024_H: list[int] = horizons_excluding(SHIFT, (2024,))
PLACEBO_DNK_DROP_H: list[int] = horizons_excluding(PLACEBO, (1921, 1922))


# ---------------------------------------------------------------------------
# Row-level helpers
# ---------------------------------------------------------------------------

def _cov_cols(horizons: Sequence[int]) -> list[str]:
    return [f"coverage95_h{int(k)}" for k in horizons]


def _errors(df: pd.DataFrame) -> pd.Series:
    if "error" in df.columns:
        return df["error"].notna()
    return pd.Series(False, index=df.index)


def valid_mask(df: pd.DataFrame, horizons: Sequence[int] | None = None) -> pd.Series:
    """error is null AND the target column(s) are finite."""
    err = _errors(df)
    if horizons is None:
        finite = np.isfinite(df["coverage_95"].to_numpy(dtype=float))
    else:
        cols = _cov_cols(horizons)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"per-horizon coverage columns missing: {missing}")
        finite = np.isfinite(df[cols].to_numpy(dtype=float)).all(axis=1)
    return (~err) & pd.Series(finite, index=df.index)


def classify_errors(err: pd.Series) -> dict[str, int]:
    """Counts by class, using scripts/final_qa.py's regexes (one source)."""
    s = err.dropna().astype(str)
    if s.empty:
        return {"machine": 0, "design_floor": 0, "method": 0, "other": 0}
    machine = s.str.contains(MACHINE)
    structural = s.str.contains(STRUCTURAL) & ~machine
    method = s.str.contains(METHOD) & ~machine & ~structural
    other = ~(machine | structural | method)
    return {"machine": int(machine.sum()), "design_floor": int(structural.sum()),
            "method": int(method.sum()), "other": int(other.sum())}


def _flag(g: pd.DataFrame, col: str, fallback: bool) -> bool:
    if col in g.columns:
        return bool(g[col].fillna(False).astype(bool).any())
    return fallback


def _f(x) -> float | None:
    x = float(x)
    return x if np.isfinite(x) else None


def arm_leaf(g: pd.DataFrame, horizons: Sequence[int] | None = None) -> dict:
    """One leaf for the rows `g` of a single family/mechanism."""
    if g.empty:
        raise ValueError("arm_leaf on an empty frame")
    model = str(g["model"].iloc[0])
    mech = str(g["mechanism"].iloc[0])
    valid = valid_mask(g, horizons)
    err = _errors(g)
    ok = g[valid]
    n = int(valid.sum())
    if horizons is None:
        cov = _f(ok["coverage_95"].mean()) if n else None
        joint = _f(ok["joint_path_coverage_95"].mean()) if n else None
        extras: dict = {}
    else:
        per_row = ok[_cov_cols(horizons)].mean(axis=1)
        cov = _f(per_row.mean()) if n else None
        joint = None
        lb = _f(ok["joint_path_coverage_95"].mean()) if n else None
        extras = {"horizons": [int(h) for h in horizons],
                  "joint_path_coverage_95_lower_bound": lb}
    ages = ok["n_ages_scored"].dropna() if "n_ages_scored" in ok.columns else pd.Series([], dtype=float)
    ages = ages.astype(int)
    leaf = {
        "coverage_95": cov,
        "joint_path_coverage_95": joint,
        "n_cells": n,
        "n_error_rows": int(err.sum()),
        "n_ages_scored": int(ages.mode().iloc[0]) if len(ages) else None,
        "conformal": _flag(g, "scores_secondary", mech in CONFORMAL_MECHANISMS),
        "grid_secondary": _flag(g, "grid_secondary", (model, mech) in SECONDARY),
    }
    if ages.nunique() > 1:
        leaf["n_ages_scored_values"] = sorted(int(v) for v in ages.unique())
    leaf.update(extras)
    return leaf


def _arm_col(df: pd.DataFrame) -> pd.Series:
    return df["model"].astype(str) + "/" + df["mechanism"].astype(str)


def _cell_col(df: pd.DataFrame) -> pd.Series:
    return (df["pop"].astype(str) + "|" + df["sex"].astype(str)
            + "|" + df["origin"].astype(int).astype(str))


def slice_table(df: pd.DataFrame, horizons: Sequence[int] | None = None,
                meta: Mapping | None = None) -> dict:
    """{_meta, 'FAM/MECH': leaf, ...} over the rows in `df`."""
    err = _errors(df)
    out: dict = {META: {
        "n_rows": int(len(df)),
        "n_cells": int(_cell_col(df).nunique()) if len(df) else 0,
        "n_error_rows": int(err.sum()),
        "error_classes": classify_errors(df["error"] if "error" in df.columns
                                         else pd.Series([], dtype=object)),
        "populations": sorted(df["pop"].astype(str).unique().tolist()),
    }}
    if horizons is not None:
        out[META]["horizons"] = [int(h) for h in horizons]
        out[META]["notes"] = [
            "coverage_95 = mean of the retained coverage95_h* columns; "
            "joint_path_coverage_95 over a horizon subset is not recoverable "
            "from runner rows (only the full-path indicator is emitted) and is "
            "null; joint_path_coverage_95_lower_bound is the full-path value."]
    if meta:
        out[META].update(dict(meta))
    for (m, u), g in df.groupby(["model", "mechanism"], sort=True):
        out[f"{m}/{u}"] = arm_leaf(g, horizons)
    return out


def _pops(df: pd.DataFrame, pops: Iterable[str], keep: bool = True) -> pd.DataFrame:
    mask = df["pop"].isin(list(pops))
    return df[mask] if keep else df[~mask]


# ---------------------------------------------------------------------------
# The four blocks
# ---------------------------------------------------------------------------

def placebo_strata(df: pd.DataFrame) -> dict:
    """Addendum 1 §A: pooled + the three registered strata."""
    out = {"pooled": slice_table(df, meta={"populations_registered": list(PLACEBO_POPS)})}
    for name, pops in PLACEBO_STRATA.items():
        out[name] = slice_table(_pops(df, pops), meta={"populations_registered": list(pops)})
    return out


def placebo_sensitivities(df: pd.DataFrame) -> dict:
    """Addendum 1 §A sensitivities 1-3 (+ the uncensored reference)."""
    neutral = PLACEBO_STRATA["neutral"]
    dnk = _pops(df, ("DNK",))
    return {
        "full_panel": slice_table(df, meta={"populations_registered": list(PLACEBO_POPS)}),
        "drop_GBR_SCO": slice_table(_pops(df, ("GBR_SCO",), keep=False),
                                    meta={"dropped": ["GBR_SCO"]}),
        "neutral_only": slice_table(_pops(df, neutral),
                                    meta={"populations_registered": list(neutral)}),
        "dnk_full_window": slice_table(dnk, meta={"test_years": list(PLACEBO.test_years)}),
        "dnk_drop_1921_1922": slice_table(
            dnk, horizons=PLACEBO_DNK_DROP_H,
            meta={"test_years": [y for y in PLACEBO.test_years if y not in (1921, 1922)],
                  "dropped_test_years": [1921, 1922]}),
    }


def shift_sensitivities(df: pd.DataFrame) -> dict:
    """Addendum 1 §B: S-S1 drop-2024, S-S2 drop USA+CHL, S-S3 register vs census."""
    return {
        "full_panel": slice_table(df, meta={"populations_registered": list(SHIFT_POPS)}),
        "drop_2024": slice_table(
            df, horizons=SHIFT_DROP_2024_H,
            meta={"test_years": [y for y in SHIFT.test_years if y != 2024],
                  "dropped_test_years": [2024]}),
        "drop_USA_CHL": slice_table(_pops(df, SHIFT_DROP_POPS, keep=False),
                                    meta={"dropped": list(SHIFT_DROP_POPS)}),
        "register_based": slice_table(_pops(df, REGISTER_BASED),
                                      meta={"populations_registered": list(REGISTER_BASED),
                                            "note": REGISTER_CENSUS_NOTE}),
        "census_based": slice_table(_pops(df, CENSUS_BASED),
                                    meta={"populations_registered": list(CENSUS_BASED),
                                          "note": REGISTER_CENSUS_NOTE}),
    }


def _primary_arms(df: pd.DataFrame) -> list[str]:
    arm = _arm_col(df)
    if "grid_secondary" in df.columns:
        sec = df["grid_secondary"].fillna(False).astype(bool)
    else:
        sec = pd.Series([(m, u) in SECONDARY for m, u in zip(df["model"], df["mechanism"])],
                        index=df.index)
    return sorted(arm[~sec].unique().tolist())


def effective_clusters(df: pd.DataFrame, registered_pops: Sequence[str]) -> dict:
    """Addendum 3 §4: per origin, populations with valid rows — per arm and panel-wide."""
    out: dict = {}
    for origin, g in df.groupby("origin", sort=True):
        g = g.assign(_arm=_arm_col(g), _valid=valid_mask(g))
        sl = slice_table(g)
        primary = _primary_arms(g)
        ok = g[g["_valid"]]
        # populations in which EVERY primary arm has a valid row for every
        # sex that population carries = the clusters a full-grid contrast
        # (addendum 3 §11) would keep
        n_sex = g.groupby("pop")["sex"].nunique()
        full = (ok[ok["_arm"].isin(primary)]
                .groupby("pop")
                .apply(lambda x: x.groupby("_arm")["sex"].nunique().reindex(primary).fillna(0)
                       .eq(n_sex[x.name]).all(), include_groups=False)
                if len(ok) else pd.Series([], dtype=bool))
        all_primary = sorted(full[full].index.astype(str).tolist()) if len(full) else []
        # the runner stores "<ExcType>: <message>", so match inside the string
        err = g["error"] if "error" in g.columns else pd.Series([None] * len(g), index=g.index)
        floor = err.fillna("").astype(str).str.contains("inadmissible: n_train", regex=False)
        sl[META].update({
            "origin": int(origin),
            "n_populations_registered": len(registered_pops),
            "n_populations_with_rows": int(g["pop"].nunique()),
            "n_clusters_any_valid_row": int(ok["pop"].nunique()),
            "populations_any_valid_row": sorted(ok["pop"].astype(str).unique().tolist()),
            "n_clusters_all_primary_arms": len(all_primary),
            "populations_all_primary_arms": all_primary,
            "n_design_floor_rows": int(floor.sum()),
            "populations_design_floor": sorted(g.loc[floor, "pop"].astype(str).unique().tolist()),
        })
        for arm, ga in g.groupby("_arm", sort=True):
            pops = sorted(ga.loc[ga["_valid"], "pop"].astype(str).unique().tolist())
            sl[arm]["n_clusters"] = len(pops)
            sl[arm]["populations"] = pops
        out[f"origin_{int(origin)}"] = sl
    return out


def _restricted_slice(d: pd.DataFrame, cells_by_arm: Mapping[str, set],
                      common: set, arms: Sequence[str], meta: dict,
                      block_of: Mapping[str, int] | None = None) -> dict:
    """Leaves for `arms` on the cell set `common`, with the §11 accounting."""
    sub = d[d["_arm"].isin(arms) & d["_cell"].isin(common)]
    sl = slice_table(sub) if len(sub) else {META: {"n_rows": 0, "n_cells": 0, "n_error_rows": 0,
                                                   "error_classes": classify_errors(pd.Series([], dtype=object)),
                                                   "populations": []}}
    sl[META].update(meta)
    sl[META]["n_cells_common"] = len(common)
    for arm in arms:
        full = cells_by_arm.get(arm, set())
        if arm not in sl:
            # arm has no rows on the common cells (its rows all sit outside)
            sl[arm] = {"coverage_95": None, "joint_path_coverage_95": None,
                       "n_cells": 0, "n_error_rows": 0, "n_ages_scored": None,
                       "conformal": arm.split("/")[1] in CONFORMAL_MECHANISMS,
                       "grid_secondary": tuple(arm.split("/")) in SECONDARY}
        leaf = sl[arm]
        leaf["n_cells_full"] = len(full)
        leaf["n_cells_lost"] = len(full) - leaf["n_cells"]
        leaf["populations_lost"] = sorted({c.split("|")[0] for c in full - common})
        if block_of is not None:
            blk = block_of.get(arm.split("/")[0])
            leaf["age_support_block"] = blk
            if blk is not None and leaf["n_ages_scored"] not in (None, blk):
                # e.g. the pre-fix CBD x copula_conf rows scored 35 of CBD's
                # 45 ages (docs/STATUS.md 2026-08-28): visible, not averaged
                leaf["age_support_mismatch"] = True
    return sl


def common_cell_losses(df: pd.DataFrame) -> dict:
    """Addendum 3 §11: cells lost per arm under each contrast sub-grid's intersection."""
    d = df.assign(_arm=_arm_col(df), _cell=_cell_col(df), _valid=valid_mask(df))
    arms = sorted(d["_arm"].unique().tolist())
    primary = _primary_arms(d)
    all_cells = set(d["_cell"].unique())
    cells_by_arm = {a: set(g.loc[g["_valid"], "_cell"]) for a, g in d.groupby("_arm")}
    no_valid = [a for a in arms if not cells_by_arm[a]]

    def intersect(arm_set: Sequence[str]) -> set:
        sets = [cells_by_arm.get(a, set()) for a in arm_set]
        return set.intersection(*sets) if sets else set()

    # age-support blocks: a family's modal n_ages_scored over its valid rows
    ok = d[d["_valid"]]
    block_of: dict[str, int] = {}
    if "n_ages_scored" in ok.columns:
        for fam, g in ok.groupby("model"):
            a = g["n_ages_scored"].dropna().astype(int)
            if len(a):
                block_of[str(fam)] = int(a.mode().iloc[0])
    blocks: dict[int, list[str]] = {}
    for fam, n in block_of.items():
        blocks.setdefault(n, []).append(fam)
    blocks = {n: sorted(f) for n, f in sorted(blocks.items())}

    base_meta = {"n_cells_panel": len(all_cells), "primary_arms": primary,
                 "arms_with_no_valid_rows": no_valid}
    out: dict = {}
    out["full_panel"] = _restricted_slice(
        d, cells_by_arm, all_cells, arms,
        {**base_meta, "intersecting_arms": [],
         "note": "uncensored reference: every valid cell of every arm; n_cells_lost = 0"})
    out["all_primary_arms"] = _restricted_slice(
        d, cells_by_arm, intersect(primary), arms,
        {**base_meta, "intersecting_arms": primary,
         "note": "cells valid in EVERY primary arm of the grid; the intersection "
                 "is a cell set, so CBD's restricted age support does not enter "
                 "here (no averaging across arms takes place)"})
    for fam in sorted({a.split("/")[0] for a in arms}):
        fam_arms = [a for a in arms if a.split("/")[0] == fam]
        inter = [a for a in fam_arms if a in primary]
        out[f"within_family:{fam}"] = _restricted_slice(
            d, cells_by_arm, intersect(inter), fam_arms,
            {**base_meta, "intersecting_arms": inter,
             "note": "mechanism contrast within one family (analyse.py "
                     "mcs_conformal_<family>, dm_native_vs_split)"})
    for mech in sorted({a.split("/")[1] for a in arms}):
        mech_arms = [a for a in arms if a.split("/")[1] == mech]
        sl: dict = {META: {**base_meta, "age_support_blocks": {str(n): f for n, f in blocks.items()},
                           "intersecting_arms": {},
                           "n_cells_common": {},
                           "note": "family contrast within one mechanism, computed per "
                                   "age-support block: families scored on different "
                                   "age ranges are never in the same ranking"}}
        for n_ages, fams in blocks.items():
            blk_arms = [a for a in mech_arms if a.split("/")[0] in fams]
            if not blk_arms:
                continue
            inter = [a for a in blk_arms if a in primary]
            common = intersect(inter) if inter else intersect(blk_arms)
            part = _restricted_slice(d, cells_by_arm, common, blk_arms, {}, block_of)
            sl[META]["intersecting_arms"][str(n_ages)] = inter
            sl[META]["n_cells_common"][str(n_ages)] = len(common)
            for a in blk_arms:
                sl[a] = part[a]
        unblocked = [a for a in mech_arms if a.split("/")[0] not in block_of]
        for a in unblocked:                       # family with no valid rows at all
            sl[a] = _restricted_slice(d, cells_by_arm, set(), [a], {})[a]
            sl[a]["age_support_block"] = None
        out[f"within_mechanism:{mech}"] = sl
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def definitions() -> dict:
    return {
        "placebo_populations": list(PLACEBO_POPS),
        "placebo_test_years": list(PLACEBO.test_years),
        "placebo_strata": {k: list(v) for k, v in PLACEBO_STRATA.items()},
        "placebo_strata_source": "PREREGISTRATION-ADDENDUM-1 §A",
        "shift_populations": list(SHIFT_POPS),
        "shift_test_years": list(SHIFT.test_years),
        "shift_population_sets": {
            "register_based": list(REGISTER_BASED),
            "census_based": list(CENSUS_BASED),
            "source": "docs/DATA-PREREQS.md §B.2 item 9 (S-S3); PREREGISTRATION-ADDENDUM-1 §B.2",
            "note": REGISTER_CENSUS_NOTE,
        },
        "shift_drop_populations": {"populations": list(SHIFT_DROP_POPS),
                                   "source": "PREREGISTRATION-ADDENDUM-1 §B.1 (S-S2)"},
        "horizon_sets": {"shift/drop_2024": SHIFT_DROP_2024_H,
                         "placebo/dnk_drop_1921_1922": PLACEBO_DNK_DROP_H},
        "valid_row": "error is null and the target column(s) are finite; error rows "
                     "are excluded from every mean and counted (n_error_rows, "
                     "_meta.error_classes)",
        "aggregation": "unweighted mean over valid rows; one row per (pop, sex, origin) "
                       "cell and arm; arms never pooled",
        "scores": "coverage_95 and joint_path_coverage_95 only — valid for every "
                  "mechanism; conformal arms carry no proper score and no 50/80 "
                  "level (addendum 2 §3)",
        "error_classes": {"machine": MACHINE.pattern, "design_floor": STRUCTURAL.pattern,
                          "method": METHOD.pattern, "source": "scripts/final_qa.py"},
        "leaf_keys": ["coverage_95", "joint_path_coverage_95", "n_cells", "n_error_rows",
                      "n_ages_scored", "conformal", "grid_secondary"],
        "metadata_prefix": "_",
        "pending_placeholder": PENDING,
    }


def _block(df: pd.DataFrame | None, fn: Callable[[pd.DataFrame], dict]):
    return PENDING if df is None else fn(df)


def build(frames: Mapping[str, pd.DataFrame | None],
          sources: Mapping[str, Sequence[str] | None] | None = None) -> dict:
    """Assemble the full JSON document from {regime: DataFrame | None}."""
    frames = {r: frames.get(r) for r in REGIMES}
    sources = {r: (list(sources[r]) if sources and sources.get(r) else None) for r in REGIMES}
    snapshot = any(Path(p).name.startswith("_") for ps in sources.values() if ps for p in ps)
    pops = {"shift": SHIFT_POPS, "placebo": PLACEBO_POPS, "stable": SHIFT_POPS}
    doc = {
        "contract_version": CONTRACT_VERSION,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "snapshot": bool(snapshot),
        "sources": sources,
        "definitions": definitions(),
        "strata": {"placebo": _block(frames["placebo"], placebo_strata)},
        "sensitivities": {"shift": _block(frames["shift"], shift_sensitivities),
                          "placebo": _block(frames["placebo"], placebo_sensitivities)},
        "effective_clusters": {r: _block(frames[r], lambda d, r=r: effective_clusters(d, pops[r]))
                               for r in REGIMES},
        "common_cell_losses": {r: _block(frames[r], common_cell_losses) for r in REGIMES},
    }
    return jsonable(doc)


def jsonable(x):
    """numpy scalars -> python, NaN -> None, tuples/sets -> lists (allow_nan=False safe)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted(jsonable(v) for v in x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if x is pd.NA:
        return None
    return x


def load_regime(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Read `path` plus its `<stem>_gp<suffix>` sibling when present."""
    paths = [path]
    gp = path.with_name(f"{path.stem}_gp{path.suffix}")
    if gp.exists():
        paths.append(gp)
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    for col in ("pop", "sex", "origin", "model", "mechanism", "coverage_95",
                "joint_path_coverage_95"):
        if col not in df.columns:
            raise ValueError(f"{path}: required column {col!r} missing")
    return df, [str(p) for p in paths]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(ROOT / "results" / "sensitivities.json"))
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    for r in REGIMES:
        p.add_argument(f"--{r}", default=None,
                       help=f"override results/{r}.parquet (absent default -> pending; "
                            "an explicit path must exist)")
    args = p.parse_args(argv)

    frames: dict[str, pd.DataFrame | None] = {}
    sources: dict[str, list[str] | None] = {}
    for r in REGIMES:
        given = getattr(args, r)
        path = Path(given) if given else Path(args.results_dir) / f"{r}.parquet"
        if not path.exists():
            if given:
                p.error(f"--{r} {path}: not found")
            frames[r], sources[r] = None, None
            print(f"[sensitivities] {r}: {path} absent -> {PENDING}", flush=True)
            continue
        frames[r], sources[r] = load_regime(path)
        n_err = int(frames[r]["error"].notna().sum()) if "error" in frames[r] else 0
        print(f"[sensitivities] {r}: {len(frames[r])} rows, {n_err} error rows "
              f"from {sources[r]}", flush=True)

    doc = build(frames, sources)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, allow_nan=False), encoding="utf-8")
    print(f"[sensitivities] wrote {out} (snapshot={doc['snapshot']})", flush=True)
    for key in ("strata", "sensitivities", "effective_clusters", "common_cell_losses"):
        for r, blk in doc[key].items():
            if blk == PENDING:
                print(f"  {key}/{r}: {PENDING}")
            else:
                print(f"  {key}/{r}: {len(blk)} slices -> {', '.join(list(blk)[:6])}"
                      f"{', ...' if len(blk) > 6 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
