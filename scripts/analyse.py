"""Analysis stage: sweep parquet -> the pre-registered contrasts.

    uv run python scripts/analyse.py results/shift.parquet --out results/shift_analysis.json

Consumes what the runner emitted; refits nothing (docs/PIPELINE.md). Every
contrast goes through ``mortcal.inference.losses_from_rows``, which applies
the addendum 3 §11 common-cell restriction and reports the intersection, so
a mechanism that fails on short panels can never be compared against one
that does not on a different set of populations.

Contrasts, stated as sub-grid comparisons per docs/GRID.md's claims
discipline — never full-factorial main effects over a ragged grid:

* ``mcs_classical_native``  — the five classical families under their own
  predictive law.
* ``mcs_conformal_<family>`` — the three conformal mechanisms plus native,
  within one family.
* ``dm_native_vs_split``     — pairwise Diebold-Mariano, per family.
* coverage tables (H2/H4) and joint-vs-marginal (H3) are descriptive
  summaries, not tests: the registered tests are the two procedures above.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mortcal.inference import (dm_wild_cluster, losses_from_rows,   # noqa: E402
                               model_confidence_set)
from mortcal.runner import CONFORMAL_MECHANISMS                     # noqa: E402

CLASSICAL = ("LC", "PLC", "CBD", "RH", "SVAR")
#: loss for any contrast that includes a conformal arm (addendum 2 §3): the
#: per-horizon Winkler/interval score at the construction level.
INTERVAL_LOSS = "winkler95"


def _mcs(df, arms, alpha, n_boot, loss, seed, ragged=False):
    """One MCS over `arms`, or a reason string if it cannot be formed.

    A ragged age support is a REASON TO SKIP, not something to work around:
    the per-horizon losses are means over each family's own scored ages
    (CBD is fit on 55+), so a contrast spanning different supports is not
    comparable. The skip reason is recorded in the output.
    """
    try:
        L, groups, names, rep = losses_from_rows(
            df, loss=loss, arms=arms, allow_ragged_age_support=ragged)
    except ValueError as exc:
        return {"skipped": str(exc)}
    out = model_confidence_set(L, groups, alpha=alpha, n_boot=n_boot,
                               names=names, rng=np.random.default_rng(seed))
    out["intersection"] = rep
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("parquet", help="sweep output from scripts/run_regime.py")
    p.add_argument("--out", required=True, help="analysis JSON path")
    p.add_argument("--loss", default="crps", choices=["crps", "logscore"])
    p.add_argument("--alpha", type=float, default=0.10, help="MCS level (registered: 90%%)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260827)
    args = p.parse_args(argv)

    df = pd.read_parquet(args.parquet)
    n_err = int(df["error"].notna().sum()) if "error" in df else 0
    print(f"[analyse] {len(df)} rows, {n_err} error rows "
          f"({100 * n_err / max(len(df), 1):.1f}%)", flush=True)

    present = {(m, u) for m, u in zip(df["model"], df["mechanism"])}
    res: dict = {
        "source": str(args.parquet),
        "loss": args.loss,
        "alpha": args.alpha,
        "n_rows": len(df),
        "n_error_rows": n_err,
        "regimes": sorted(df["regime"].unique().tolist()),
    }

    # --- 1. classical families under their own predictive law ---------------
    # Two versions: the full set (skips if CBD's restricted age support makes
    # it incomparable) and the full-age families only, which is the contrast
    # that is always valid.
    arms = [(m, "native") for m in CLASSICAL if (m, "native") in present]
    if len(arms) >= 2:
        res["mcs_classical_native"] = _mcs(df, arms, args.alpha, args.n_boot,
                                           args.loss, args.seed)
    full_age = [(m, "native") for m in CLASSICAL
                if m != "CBD" and (m, "native") in present]
    if len(full_age) >= 2:
        res["mcs_classical_native_full_age"] = _mcs(
            df, full_age, args.alpha, args.n_boot, args.loss, args.seed)

    # --- 2. mechanism contrast WITHIN each family ---------------------------
    # Conformal proper scores are flagged secondary (addendum 2 §3): rank them
    # only against each other, never against distributional mechanisms.
    for fam in sorted({m for m, _ in present}):
        conf = [(fam, u) for u in sorted(CONFORMAL_MECHANISMS)
                if (fam, u) in present]
        if len(conf) >= 2:
            # interval mechanisms are compared on the INTERVAL score: their
            # crps/logscore are flagged placeholders (losses_from_rows raises)
            res[f"mcs_conformal_{fam}"] = _mcs(df, conf, args.alpha,
                                               args.n_boot, INTERVAL_LOSS, args.seed)

    # --- 3. pairwise DM: native vs split conformal, per family --------------
    dm = {}
    for fam in sorted({m for m, _ in present}):
        pair = [(fam, "native"), (fam, "split_conf")]
        if not all(c in present for c in pair):
            continue
        try:
            # native vs conformal: mixed arms -> interval score, not crps
            L, groups, names, rep = losses_from_rows(df, loss=INTERVAL_LOSS, arms=pair)
        except ValueError as exc:
            dm[fam] = {"skipped": str(exc)}
            continue
        a, b = names.index(f"{fam}/native"), names.index(f"{fam}/split_conf")
        out = dm_wild_cluster(L[:, a], L[:, b], groups, n_boot=4999,
                              rng=np.random.default_rng(args.seed))
        out["arms"] = [names[a], names[b]]
        out["intersection"] = rep
        dm[fam] = out
    if dm:
        res["dm_native_vs_split"] = dm

    # --- 4. descriptive calibration tables (H2, H3, H4) ---------------------
    ok = df[df["error"].isna()] if "error" in df else df
    if len(ok):
        agg = (ok.groupby(["model", "mechanism"])
                 .agg(n_cells=("coverage_95", "size"),
                      coverage_95=("coverage_95", "mean"),
                      coverage_80=("coverage_80", "mean"),
                      coverage_50=("coverage_50", "mean"),
                      joint_95=("joint_path_coverage_95", "mean"),
                      winkler_95=("winkler_95", "mean"),
                      crps=("crps_logmx", "mean"),
                      pit_ks=("pit_ks_stat", "mean"))
                 .reset_index())
        agg["joint_minus_marginal"] = agg["joint_95"] - agg["coverage_95"]
        res["calibration_table"] = json.loads(agg.to_json(orient="records"))
        bands = [c for c in ok.columns if c.startswith("coverage_95_band")]
        if bands:
            res["coverage_by_age_band"] = json.loads(
                ok.groupby(["model", "mechanism"])[bands].mean()
                  .reset_index().to_json(orient="records"))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[analyse] wrote {args.out}", flush=True)
    for k, v in res.items():
        if isinstance(v, dict) and "in_set" in v:
            print(f"  {k}: in_set={v['in_set']} "
                  f"(cells kept {v['intersection']['n_cells_kept']}, "
                  f"dropped {v['intersection']['n_cells_dropped']})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
