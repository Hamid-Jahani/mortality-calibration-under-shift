"""Real-data sweep CLI for the pre-registered regimes.

    uv run --no-sync python scripts/run_regime.py shift   --out results/shift_classical.parquet
    uv run --no-sync python scripts/run_regime.py placebo --out results/placebo_classical.parquet
    uv run --no-sync python scripts/run_regime.py stable  --out results/stable_classical.parquet

Passes allow_real=True: that flag is the auditable act of running on HMD
data, permitted only once validation gates 1-3 pass (they do — see
docs/STATUS.md) and the verifier ledgers are closed. Results are written
incrementally by run_regime so a killed run keeps its completed cells.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mortcal.data import build_panel                     # noqa: E402
from mortcal import runner, splits                       # noqa: E402

DS = ROOT / "Dataset"
DEATHS = DS / "deaths" / "Deaths_1x1" / "Deaths_1x1.txt"
EXPOS = DS / "exposures" / "Exposures_1x1" / "Exposures_1x1.txt"

REGIMES = {"shift": splits.SHIFT, "placebo": splits.PLACEBO, "stable": splits.STABLE}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("regime", choices=sorted(REGIMES))
    p.add_argument("--out", required=True, help="parquet path (written incrementally)")
    p.add_argument("--models", default="all", help="comma list of registry names or 'all'")
    p.add_argument("--mechanisms", default="all", help="comma list of registry names or 'all'")
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--pops", default=None, help="optional comma subset of the regime's populations")
    p.add_argument("--seed", type=int, default=20260825)
    args = p.parse_args(argv)

    regime = REGIMES[args.regime]
    regimes = list(regime) if isinstance(regime, tuple) else [regime]
    pops = tuple(args.pops.split(",")) if args.pops else regimes[0].pops
    if args.pops:
        regimes = [r.__class__(**{**r.__dict__, "pops": pops}) for r in regimes]

    models = list(runner.MODELS) if args.models == "all" else args.models.split(",")
    mechs = list(runner.MECHANISMS) if args.mechanisms == "all" else args.mechanisms.split(",")

    t0 = time.time()
    print(f"[run_regime] loading panel for {len(pops)} populations ...", flush=True)
    panel = build_panel(DEATHS, EXPOS, pops=list(pops), age_max=99)
    print(f"[run_regime] panel rows={len(panel):,}  load={time.time() - t0:.0f}s", flush=True)
    print(f"[run_regime] regime={args.regime} origins={len(regimes)} models={models} mechanisms={mechs}", flush=True)

    # run_regime writes its parquet once at the END of a sweep, so a killed
    # multi-hour run would lose everything. We therefore sweep one population
    # at a time into results/<name>.parts/<POP>.parquet (skipping parts that
    # already exist -> resumable) and concatenate into --out at the end.
    out = Path(args.out)
    parts_dir = out.with_suffix(".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)  # noqa: E731

    import pandas as pd  # local: keep module import cheap for --help

    for pop in pops:
        part = parts_dir / f"{pop}.parquet"
        if part.exists():
            log(f"{pop}: part exists, skipping (resume)")
            continue
        pop_regimes = [r.__class__(**{**r.__dict__, "pops": (pop,)}) for r in regimes]
        sub = panel[panel["pop"] == pop]
        log(f"{pop}: {len(sub):,} panel rows, {len(pop_regimes)} origin(s)")
        df_pop = runner.run_regime(
            sub, pop_regimes, models, mechs, n_samples=args.n_samples,
            out_path=part, allow_real=True, base_seed=args.seed, log=log,
        )
        n_err = int(df_pop["error"].notna().sum()) if "error" in df_pop else 0
        log(f"{pop}: rows={len(df_pop)} error_rows={n_err}")

    parts = sorted(parts_dir.glob("*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df.to_parquet(out, index=False)
    n_err = int(df["error"].notna().sum()) if "error" in df else 0
    print(f"[run_regime] done: rows={len(df)} error_rows={n_err} parts={len(parts)} "
          f"elapsed={time.time() - t0:.0f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
