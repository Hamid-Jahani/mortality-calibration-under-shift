"""Real-data sweep CLI for the pre-registered regimes.

    python scripts/run_regime.py shift   --out results/shift.parquet   --jobs 12 --exclude-models GP
    python scripts/run_regime.py shift   --out results/shift_gp.parquet --models GP --jobs 2
    python scripts/run_regime.py placebo --out results/placebo.parquet --jobs 12 --exclude-models GP
    python scripts/run_regime.py stable  --out results/stable.parquet  --jobs 12 --exclude-models GP

Passes allow_real=True: that flag is the auditable act of running on HMD
data, permitted only once validation gates 1-3 pass (they do — see
docs/STATUS.md) and the verifier ledgers are closed.

Cells: every worker runs only the (model, mechanism) pairs that are
ADMISSIBLE per docs/GRID.md (mortcal.runner.ADMISSIBLE) — the runner
validates pairs by RAISING, so handing it the full mechanism list for every
family kills the sweep on the first inadmissible cell (measured 2026-08-27:
(LC, ensemble)). The four "(s)" secondary cells are excluded unless
--include-secondary is passed; they are run-if-time by design.

Resumable: parts land in results/<name>.parts/<POP>__<MODEL>.parquet and any
existing part is skipped, so a killed run resumes at (population, model)
granularity. The final --out parquet is the concatenation of all parts.
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


def _worker_init() -> None:
    """Pin every numeric library in this worker to one thread."""
    import os
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def admissible_pairs(models, mechs, include_secondary: bool) -> dict[str, list[str]]:
    """model -> mechanisms to run, restricted to docs/GRID.md admissibility."""
    out: dict[str, list[str]] = {}
    for m in models:
        keep = [u for u in mechs
                if (m, u) in runner.ADMISSIBLE
                and (include_secondary or (m, u) not in runner.SECONDARY)]
        if keep:
            out[m] = keep
    return out


def _part_name(pop: str, model: str, tag: str) -> str:
    """Part file for (population, model[, origin subset]). The origin tag keeps
    two processes that split one regime's origins (e.g. --origins on a second
    machine) from overwriting each other's parts."""
    return f"{pop}__{model}{tag}.parquet"


def _run_one_pop(pop, parts_dir, sub, regimes, pairs, n_samples, seed, log, tag="") -> str:
    """All admissible cells of one population; one part per model; resumable."""
    pop_regimes = [r.__class__(**{**r.__dict__, "pops": (pop,)}) for r in regimes]
    parts_dir = Path(parts_dir)
    rows = errs = 0
    for model, mechs in pairs.items():
        part = parts_dir / _part_name(pop, model, tag)
        if part.exists():
            continue
        df = runner.run_regime(
            sub, pop_regimes, [model], mechs, n_samples=n_samples,
            out_path=part, allow_real=True, base_seed=seed, log=log,
        )
        rows += len(df)
        errs += int(df["error"].notna().sum()) if "error" in df else 0
    return f"{pop}: rows={rows} error_rows={errs}"


def _run_one_pop_task(task) -> str:
    pop, parts_dir, sub, regimes, pairs, n_samples, seed, tag = task
    t0 = time.time()
    quiet = lambda m: None  # noqa: E731 — per-cell chatter stays in the worker
    try:
        msg = _run_one_pop(pop, parts_dir, sub, regimes, pairs, n_samples, seed, quiet, tag)
    except Exception as exc:  # noqa: BLE001 — a population must never kill the pool
        msg = f"{pop}: FAILED {type(exc).__name__}: {exc}"
    return f"{msg} ({time.time() - t0:.0f}s)"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("regime", choices=sorted(REGIMES))
    p.add_argument("--out", required=True, help="final parquet path (concatenation of parts)")
    p.add_argument("--models", default="all", help="comma list of registry names or 'all'")
    p.add_argument("--mechanisms", default="all", help="comma list of registry names or 'all'")
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--pops", default=None, help="optional comma subset of the regime's populations")
    p.add_argument("--origins", default=None,
                   help="comma list of training-cutoff years (STABLE only) so one regime can be "
                        "split across processes/machines; parts are then tagged __o<first>-<last>")
    p.add_argument("--exclude-models", default="", help="comma list of registry names to skip "
                   "(e.g. GP, so the 1.6 GB GP cells run in a separate low-parallel pass)")
    p.add_argument("--include-secondary", action="store_true",
                   help="also run the four docs/GRID.md '(s)' cells (run-if-time)")
    p.add_argument("--jobs", type=int, default=1, help="populations processed in parallel; each "
                   "worker pins BLAS/torch to ONE thread so 12 jobs on 12 cores do not oversubscribe")
    p.add_argument("--seed", type=int, default=20260825)
    args = p.parse_args(argv)

    regime = REGIMES[args.regime]
    regimes = list(regime) if isinstance(regime, tuple) else [regime]
    pops = tuple(args.pops.split(",")) if args.pops else regimes[0].pops
    if args.pops:
        regimes = [r.__class__(**{**r.__dict__, "pops": pops}) for r in regimes]
    tag = ""
    if args.origins:
        want = {int(y) for y in args.origins.split(",")}
        regimes = [r for r in regimes if r.train_max_year in want]
        if not regimes:
            raise SystemExit(f"no origins of {args.regime} match --origins {args.origins}")
        ys = sorted(r.train_max_year for r in regimes)
        tag = f"__o{ys[0]}-{ys[-1]}"

    models = list(runner.MODELS) if args.models == "all" else args.models.split(",")
    excluded = {m for m in args.exclude_models.split(",") if m}
    models = [m for m in models if m not in excluded]
    mechs = list(runner.MECHANISMS) if args.mechanisms == "all" else args.mechanisms.split(",")
    pairs = admissible_pairs(models, mechs, args.include_secondary)
    n_cells = sum(len(v) for v in pairs.values())

    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)  # noqa: E731
    log(f"loading panel for {len(pops)} populations ...")
    panel = build_panel(DEATHS, EXPOS, pops=list(pops), age_max=99)
    log(f"panel rows={len(panel):,}")
    log(f"regime={args.regime} origins={len(regimes)} cells/pop={n_cells} "
        f"({'incl.' if args.include_secondary else 'excl.'} secondary): "
        + "; ".join(f"{m}:{','.join(u)}" for m, u in pairs.items()))

    out = Path(args.out)
    parts_dir = out.with_suffix(".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd  # local: keep module import cheap for --help

    todo = [pop for pop in pops
            if any(not (parts_dir / _part_name(pop, m, tag)).exists() for m in pairs)]
    skipped = [pop for pop in pops if pop not in todo]
    if skipped:
        log(f"{len(skipped)} population(s) complete, skipping (resume): {','.join(skipped)}")

    jobs = max(1, min(args.jobs, len(todo))) if todo else 1
    log(f"{len(todo)} population(s) to run, jobs={jobs}")
    if jobs == 1:
        for pop in todo:
            log(_run_one_pop(pop, parts_dir, panel[panel["pop"] == pop], regimes, pairs,
                             args.n_samples, args.seed, log, tag))
    else:
        # one process per population; each worker pins BLAS/torch to a single
        # thread (see _worker_init) so `jobs` workers on `jobs` cores do not
        # oversubscribe — the 2.2x inflation measured in results/timings.json
        # came from overlapping multi-threaded runs.
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        tasks = [(pop, str(parts_dir), panel[panel["pop"] == pop], regimes, pairs,
                  args.n_samples, args.seed, tag) for pop in todo]
        with ctx.Pool(jobs, initializer=_worker_init) as pool:
            for msg in pool.imap_unordered(_run_one_pop_task, tasks):
                log(msg)

    parts = sorted(parts_dir.glob("*__*.parquet"))
    if not parts:
        log("no parts written")
        return 1
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df.to_parquet(out, index=False)
    n_err = int(df["error"].notna().sum()) if "error" in df else 0
    log(f"done: rows={len(df)} error_rows={n_err} parts={len(parts)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
