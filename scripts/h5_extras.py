"""H5 tail-shortfall shares + H1 rank-correlation cluster-bootstrap CI.

    python scripts/h5_extras.py [--parquet results/shift.parquet]
        [--parquet results/shift_gp.parquet] [--out results/h5_extras.json]

Two descriptive quantities the generated tables do not carry, both computed
from the QA-passed shift parquets and written to ``results/h5_extras.json``
(nothing is refit, nothing re-tested):

1. **Annuity tail shortfall** (paper §7.2). For every distributional,
   non-secondary (family, mechanism) arm: the share of admissible cells in
   which the realised annuity factor exceeds the model's own 97.5% sample
   quantile (``ann65_obs > ann65_q975``), and the mean relative shortfall
   ``(ann65_obs - ann65_q975) / ann65_point`` over exactly those cells;
   symmetrically for the lower side (``ann65_obs < ann65_q025``, excess
   ``(ann65_q025 - ann65_obs) / ann65_point``). Admissible cell = row with
   no ``error``, ``scores_secondary`` False (conformal arms have filler
   samples; addendum 2 §3), ``grid_secondary`` False, and finite
   obs/point/quantile columns; ``n`` per arm is reported.

2. **H1 rank-correlation CI** (paper §6, tab-h1 note). A cluster bootstrap
   over populations (the 20 correlated clusters of the inference plan) of
   the two Spearman correlations the tab-h1 main block reports: rank by
   mean RMSE on log rates vs rank by mean CRPS on log rates, and vs rank by
   mean Poisson log score. The statistic is computed exactly as
   ``make_tables._rank_block`` computes it -- common-cell restriction
   (addendum 3 §11) over the full-age distributional arms, unweighted mean
   per (population, sex, origin) row, ``rank(method="min")``, Spearman on
   the rank vectors. Each replicate resamples the kept populations with
   replacement (a pairs cluster bootstrap, B=2000, seed 20260830) and
   recomputes both correlations; the 2.5/97.5 percentiles are written to
   the JSON. Deterministic given the seed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_tables as mt  # noqa: E402

H1_BOOT_B = 2000
H1_BOOT_SEED = 20260830


# ---------------------------------------------------------------------------
# (1) annuity tail shortfall per arm
# ---------------------------------------------------------------------------

def annuity_shortfall(df: pd.DataFrame, regime: str = "shift") -> list[dict]:
    """Per distributional non-secondary arm: 97.5% / 2.5% exceedance shares
    of the realised annuity factor and the conditional mean relative excess."""
    sub = mt.regime_frame(df, regime)
    ok = mt.distributional(mt.valid_rows(sub))
    if "grid_secondary" in ok:
        ok = ok[~ok["grid_secondary"].fillna(False).astype(bool)]
    err = sub[sub["error"].notna() & ~sub["scores_secondary"]]
    out = []
    arms = mt.sort_cells(set(zip(ok["model"], ok["mechanism"])))
    for m, u in arms:
        a = ok[(ok["model"] == m) & (ok["mechanism"] == u)]
        n_err = int(((err["model"] == m) & (err["mechanism"] == u)).sum())
        obs, pt = a["ann65_obs"], a["ann65_point"]
        lo, hi = a["ann65_q025"], a["ann65_q975"]
        fin = obs.notna() & pt.notna() & lo.notna() & hi.notna() & (pt != 0)
        obs, pt, lo, hi = obs[fin], pt[fin], lo[fin], hi[fin]
        rec = {"model": m, "mechanism": u,
               "n": int(len(a)), "n_admissible": int(fin.sum()), "n_err": n_err}
        for side, mask, excess in (
                ("upper", obs > hi, (obs - hi) / pt),
                ("lower", obs < lo, (lo - obs) / pt)):
            k = int(mask.sum())
            rec[f"{side}_exceed_n"] = k
            rec[f"{side}_exceed_share"] = float(mask.mean()) if len(obs) else None
            rec[f"{side}_mean_rel_shortfall"] = (
                float(excess[mask].mean()) if k else None)
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# (2) cluster bootstrap of the tab-h1 Spearman correlations
# ---------------------------------------------------------------------------

def _rho_pair(st: pd.DataFrame) -> dict[str, float]:
    """The two correlations exactly as make_tables._rank_block reports them."""
    rho = {}
    for c in ("crps_logmx", "poisson_log_score"):
        a = st["rmse_logmx"].rank(method="min").astype(int)
        b = st[c].rank(method="min").astype(int)
        rho[c] = float(a.corr(b, method="spearman"))
    return rho


def h1_bootstrap(df: pd.DataFrame, regime: str = "shift",
                 B: int = H1_BOOT_B, seed: int = H1_BOOT_SEED) -> dict:
    sub = mt.regime_frame(df, regime)
    ok = mt.distributional(mt.valid_rows(sub))
    all_arms = mt.sort_cells(set(zip(ok["model"], ok["mechanism"])))
    main_arms = [a for a in all_arms if a[0] not in mt.RESTRICTED_AGE_FAMILIES]

    # point estimates from the very function tab-h1 uses
    res, rep = mt._rank_block(sub[~sub["scores_secondary"]], main_arms)
    if res is None:
        raise SystemExit("no common cell across the distributional full-age arms")
    st_point, rho_point = res

    # base frame: valid rows of the main arms on the kept common units
    kept, _ = mt.common_cells(sub[~sub["scores_secondary"]], main_arms)
    base = mt.restrict(mt.valid_rows(sub[~sub["scores_secondary"]]), kept)
    base = base[[(m, u) in set(main_arms)
                 for m, u in zip(base["model"], base["mechanism"])]]

    pops = sorted(base["pop"].unique())
    n_pops = len(pops)
    arm_ix = {a: i for i, a in enumerate(main_arms)}
    pop_ix = {p: i for i, p in enumerate(pops)}

    # per (arm, pop): sum of each score and row count; a resample's mean is
    # then the count-weighted mean, identical to concatenating the drawn
    # populations' rows and taking groupby(...).mean()
    S = {c: np.zeros((len(main_arms), n_pops)) for c in mt.H1_COLS}
    N = np.zeros((len(main_arms), n_pops))
    for (m, u, p), g in base.groupby(["model", "mechanism", "pop"]):
        i, j = arm_ix[(m, u)], pop_ix[p]
        N[i, j] = len(g)
        for c in mt.H1_COLS:
            S[c][i, j] = g[c].sum()

    # cross-check: the all-ones weighting must reproduce the point estimate
    w1 = np.ones(n_pops)
    for c in mt.H1_COLS:
        full = S[c] @ w1 / (N @ w1)
        ref = (st_point.set_index(["model", "mechanism"])
               .loc[main_arms, c].to_numpy())
        assert np.allclose(full, ref, rtol=0, atol=1e-12), c

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_pops, size=(B, n_pops))
    rhos = {c: np.empty(B) for c in ("crps_logmx", "poisson_log_score")}
    for b in range(B):
        w = np.bincount(draws[b], minlength=n_pops).astype(float)
        st = pd.DataFrame({c: S[c] @ w / (N @ w) for c in mt.H1_COLS})
        pair = _rho_pair(st)
        for c in rhos:
            rhos[c][b] = pair[c]

    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    return {
        "B": B, "seed": seed, "cluster": "population",
        "n_pops": n_pops, "pops": pops,
        "n_arms": len(main_arms),
        "n_units_kept": int(rep["n_kept"]),
        "n_units_total": int(rep["n_units_total"]),
        "rho_rmse_crps": {"point": round(rho_point["crps_logmx"], 6),
                          "ci95": ci(rhos["crps_logmx"])},
        "rho_rmse_logscore": {"point": round(rho_point["poisson_log_score"], 6),
                              "ci95": ci(rhos["poisson_log_score"])},
        "n_nan_replicates": {c: int(np.isnan(v).sum()) for c, v in rhos.items()},
    }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parquet", action="append", default=None,
                    help="runner parquet(s); default results/shift.parquet + results/shift_gp.parquet")
    ap.add_argument("--out", default=None,
                    help="default results/h5_extras.json, or "
                         "results/h5_extras_<regime>.json for a non-shift regime")
    # The regime was hardcoded to "shift" until 2026-09-04, so the stable-control
    # figures quoted in the actuarial section could not be regenerated from the
    # command line at all -- they had to be produced by importing the function.
    ap.add_argument("--regime", default="shift",
                    help="regime to summarise; also picks the default parquets")
    ap.add_argument("--B", type=int, default=H1_BOOT_B)
    ap.add_argument("--seed", type=int, default=H1_BOOT_SEED)
    args = ap.parse_args(argv)

    paths = args.parquet or [str(ROOT / "results" / f"{args.regime}.parquet"),
                             str(ROOT / "results" / f"{args.regime}_gp.parquet")]
    out_path = args.out or str(
        ROOT / "results" / ("h5_extras.json" if args.regime == "shift"
                            else f"h5_extras_{args.regime}.json"))
    df = mt.load_rows(paths)

    shortfall = annuity_shortfall(df, regime=args.regime)
    boot = h1_bootstrap(df, regime=args.regime, B=args.B, seed=args.seed)

    payload = {
        "script": "scripts/h5_extras.py",
        "inputs": [Path(p).name for p in paths],
        "regime": args.regime,
        "annuity_shortfall": {
            "definition": {
                "upper_exceed_share": "share of admissible cells with ann65_obs > ann65_q975",
                "upper_mean_rel_shortfall": "mean of (ann65_obs - ann65_q975)/ann65_point over those cells",
                "lower_exceed_share": "share of admissible cells with ann65_obs < ann65_q025",
                "lower_mean_rel_shortfall": "mean of (ann65_q025 - ann65_obs)/ann65_point over those cells",
                "admissible": "no error, scores_secondary False, grid_secondary False, finite obs/point/q025/q975",
            },
            "arms": shortfall,
        },
        "h1_rank_correlation": boot,
    }
    out = Path(out_path)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    hdr = (f"{'family':6s} {'mech':9s} {'n_adm':>5s} {'n_err':>5s} "
           f"{'P(obs>q975)':>11s} {'rel short':>10s} {'P(obs<q025)':>11s} {'rel short':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for r in shortfall:
        def _f(v, nd=3):
            return "--" if v is None else f"{v:.{nd}f}"
        print(f"{r['model']:6s} {r['mechanism']:9s} {r['n_admissible']:5d} {r['n_err']:5d} "
              f"{_f(r['upper_exceed_share']):>11s} {_f(r['upper_mean_rel_shortfall']):>10s} "
              f"{_f(r['lower_exceed_share']):>11s} {_f(r['lower_mean_rel_shortfall']):>10s}")
    print()
    for k in ("rho_rmse_crps", "rho_rmse_logscore"):
        v = boot[k]
        print(f"{k}: point {v['point']:.4f}, 95% cluster-bootstrap CI "
              f"[{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] "
              f"(B={boot['B']}, seed={boot['seed']}, {boot['n_pops']} populations)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
