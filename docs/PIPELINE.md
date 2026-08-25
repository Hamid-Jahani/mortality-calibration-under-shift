# PIPELINE — running the pre-registered evaluation

How results tables are produced once the validation gates close. The runner is
`src/mortcal/runner.py`; the pre-registered regimes are `src/mortcal/splits.py`
(a transcription of PREREGISTRATION.md — editing either is a reportable
protocol deviation).

## Gate status (2026-08-26)

| Gate (PREREGISTRATION.md) | Status |
|---|---|
| 1. Synthetic truth (`tests/test_synthetic_calibration.py`) | **PASSES** — correctly-specified Poisson-LC attains nominal coverage against the LATENT true m_x; the harness detects an injected break as under-coverage. Known documented finding: two-stage SVD-LC is mildly overdispersed in-DGP. |
| 2. Oracle parity (Python LC/PLC vs R StMoMo, 1e-6 relative on α, β, κ) | **PASSES** (2026-08-25) — identical log-likelihood, β relative difference 2.4e-14 on SWE males 1950–2000 (`scripts/check_parity.py`, `results/parity/`). |
| 3. Life-table parity (our ex vs HMD published ex) | Spot-check harness in `tests/test_lifetable.py`; full HMD spot checks run with the first real-data pass. |
| Reading gate: Dowd et al. (2010b), Schnürch & Korn (2022) before related-work text | **OPEN** — SSRN links in `literature/GET-THESE.md`, browser-only. |
| Runner defect ledger (`docs/STATUS.md` items 1–6) | **CLOSED** 2026-08-26. |

**The runner still guards real data**: `run_regime` raises unless the regime
is named `"synthetic"` or `allow_real=True` is passed. Gate 2 has closed; the
flag is kept so that producing a real-data table is a deliberate, auditable
act rather than a default.

## What the runner does

- `MODELS` — classical families in scope for the CPU sweep: `LC`
  (Lee–Carter SVD), `PLC` (Poisson Lee–Carter), `CBD` (M5), `RH` (M2-A),
  `SVAR` (banded VAR on improvements).
- `MODEL_KWARGS` — per-family constructor arguments applied under EVERY
  mechanism. Currently `CBD: {age_min: 55}` — M5 assumes logit q_x linear in
  age, which holds at older ages only, so the fit uses ages 55–99 and returns
  NaN samples below 55. The kwarg reaches the native instance, every
  bootstrap refit (`PoissonBootstrap(**model_kwargs)`) and every conformal
  member/centre refit (`functools.partial` base factory).
- `MECHANISMS` — `native`, `pboot` (Poisson bootstrap), `split_conf`,
  `enbpi`, `copula_conf`. Admissibility is transcribed from `docs/GRID.md`;
  inadmissible or unknown cells raise `InadmissibleCellError` before any fit.
  All five families implement `fitted_mx()`, so `pboot` builds for each.
- `run_cell(D, E, model, mechanism, h, n_samples, rng, obs_D=…, obs_E=…)`
  fits one grid cell on the training matrices `[n_ages, n_train_years]`,
  draws `[n_samples, h, n_ages]` predictive m_x paths, and scores them
  through the single evaluation path (`mortcal.eval` + `mortcal.lifetable`).
- `run_regime(panel_df, regime, models, mechanisms, n_samples, out_path)`
  sweeps (pop × sex × origin × model × mechanism) and writes ONE parquet row
  per cell (pandas + pyarrow). A cell whose fit or scoring raises is skipped
  and logged; the exception string lands in the row's `error` column so the
  sweep always finishes and failures are auditable in the output itself.
  Per-cell seeds derive from the global seed 20260825 via
  `SeedSequence([base, origin, crc32(pop), crc32(sex), crc32(model),
  crc32(mech)])` and are recorded in the `seed_entropy` column (rule 7).

### Scoring conventions (fixed in `run_cell`, documented once there)

- **Age mask.** Only ages on which every predictive sample and every
  observed rate is finite are scored; `n_ages_scored` and `n_cells`
  (= h × n_ages_scored) record the denominator of every mean. For CBD that is
  ages 55–99 (45 ages). Band / by-age columns are NaN / null where nothing
  is scored.
- **Point functional.** ONE convention for every point metric: the
  pointwise MEDIAN of the predictive samples — of log m_x per cell for
  RMSE/MAE, of the per-sample life-table functionals for `e0/e65/ann65_point`.
  No column uses a sample mean.
- **Count scale.** Poisson log score on observed deaths rounded half-up
  (`round_deaths` = floor(d + 0.5); the pre-registered "nearest integer"
  for Lexis-split fractional deaths). `crps_counts` is the rounding-free
  sensitivity companion: sampled death counts against UNROUNDED observed
  deaths. PLC and RH draw counts through their own `sample_deaths`; every
  other family/wrapper composes the identical construction
  D ~ Poisson(E_future · sample_mx) in the runner.
- **Exposures.** `E_future` defaults to the observed test exposures
  (realised exposures treated as known offsets, not forecast).
- **Derived quantities under masking.** e_x and ä_x are ratios to l_x, so
  e65 / ä65 are exact from a table starting at 55; e0 is NaN whenever age 0
  is not scored (CBD). Observed values are always computed on the full panel.

### Column glossary (one parquet row per cell)

| Group | Columns |
|---|---|
| keys | `regime, pop, sex, origin, model, mechanism, h, n_samples, seed_entropy, error` |
| denominators | `n_ages_scored, n_cells` |
| point | `rmse_logmx, mae_logmx` |
| proper scores | `crps_logmx, poisson_log_score, crps_counts` |
| intervals | `coverage_{50,80,95}, winkler_{50,80,95}, joint_path_coverage_95` |
| calibration by age (H4) | `coverage_{50,80,95}_band{0_24,25_64,65_99}`, `pit_ks_band{0_24,25_64,65_99}`, `cov95_by_age` (JSON list, one entry per panel age = mean 95% hit indicator over horizons, `null` where masked) |
| PIT | `pit_ks_stat`, `pit_hist` (JSON, 10 equal bins, sums to 1) |
| Murphy | `murphy_{reliability,resolution,uncertainty,brier}` — Murphy (1973) on the 95% hit indicators with the constant forecast 0.95: reliability = (0.95 − coverage)², resolution = 0 (degenerate by construction, reported because pre-registered); `murphy_pit_{reliability,resolution,uncertainty}` — Broecker (2009) divergence form on the PIT histogram with the uniform as reference, whose reliability is the χ²-type distance from uniformity |
| per horizon (DM loss series) | `crps_h{k}, logscore_h{k}, coverage95_h{k}, winkler95_h{k}`, k = 1..H of the regime; regimes with different H in one file leave the extra horizons NaN |
| derived (H5) | `{e0,e65,ann65}_{point,q025,q975,obs,error}`, error = point − obs, from the horizon-1 sample table (ä65 at 2%) |
| flag | `scores_secondary` (True for conformal mechanisms) |

The by-age curve and the PIT histogram are persisted so the H4 figures and
the PIT panels plot straight from parquet without refits.

### Conformal cells and proper scores

The conformal wrappers emit uniform-in-interval samples (see the module
docstring of `src/mortcal/uq/conformal.py`): intervals, not distributions.
Every row carries a boolean `scores_secondary` column — `True` for
`split_conf` / `enbpi` / `copula_conf` — and CRPS / log score / PIT from
flagged rows go only to a flagged appendix table, never ranked against
distributional mechanisms.

Cost: the conformal wrappers REFIT the base family (split: 2 fits; EnbPI and
copula: K=10 trailing-block members + 1 centre refit); `pboot` fits 1 + B.
See the cost note in `docs/GRID.md`.

Known residual (conformal.py, not the runner): CBD × conformal scores ages
65–99 only (`n_ages_scored` = 35), because the NaN residuals of ages 25–54
are pooled with 55–64 inside the 25–64 Mondrian band and the band quantile
turns NaN. A nan-aware band quantile in `_conformal_quantile` restores 55–64.

## Running each regime (after gates close)

```python
import numpy as np, pandas as pd
from mortcal.data.hmd import build_panel
from mortcal.splits import REGIMES
from mortcal.runner import MODELS, MECHANISMS, run_regime

panel = build_panel("data/Deaths_1x1.txt", "data/Exposures_1x1.txt")

# Shift (primary): train <= 2019, test 2020-2024, 20 populations
run_regime(panel, REGIMES["shift"], list(MODELS), list(MECHANISMS),
           n_samples=2000, out_path="results/shift.parquet",
           allow_real=True)   # deliberate, auditable act — see guard

# Stable (control): 13 expanding origins, one Regime each
run_regime(panel, REGIMES["stable"], list(MODELS), list(MECHANISMS),
           n_samples=2000, out_path="results/stable.parquet",
           allow_real=True)

# Placebo break: train <= 1913, test 1914-1922, 11 populations
run_regime(panel, REGIMES["placebo"], list(MODELS), list(MECHANISMS),
           n_samples=2000, out_path="results/placebo.parquet",
           allow_real=True)
```

`mech_kwargs` is forwarded to every mechanism wrapper constructor (e.g.
`{"B": 200}` for `pboot`); run one mechanism per call when the wrappers
need different arguments.

## Deliberately NOT implemented yet

- **Neural families** (neural-LC, CNN-LC, LSTM-on-k_t, distributional NB
  head) and the **multi-output GP** — GPU-budget arms of the grid; the
  runner registries hold classical families only.
- **Deep ensemble / MC dropout mechanisms** — only meaningful for the neural
  arms (inadmissible for classical fits per `docs/GRID.md`), so absent from
  `MECHANISMS` entirely.
- **Inference layer** — wild cluster bootstrap over populations
  (20 correlated clusters), Diebold–Mariano on the per-horizon loss series,
  and the 90% Model Confidence Set operate on the parquet outputs
  downstream (`mortcal.inference`); not in the runner.
- **Analysis/figures** — calibration-by-age curves, PIT histograms, Murphy
  tables and H1–H5 tests consume the per-cell parquet rows; separate
  analysis stage.
- **STMF weekly robustness, 2024-provisional and age-cap sensitivities** —
  planned secondary runs (PREREGISTRATION.md "Planned robustness").

## Tests

`tests/test_runner.py` exercises the sweep end to end on a synthetic
Poisson-LC truth extended to ages 0–99 (2 populations, 2 origins, LC+PLC ×
native+split_conf), asserts every column present with the documented type,
the means equal the averages of their per-horizon series, the Murphy
identities hold, the CBD age mask lands exactly on ages 55–99 with e0 NaN by
design and e65/ä65 exact from the truncated table, `(RH, pboot)` scores with
a tiny B, per-horizon column counts equal H, the parquet round-trips
(including the JSON vectors), real regimes are refused without
`allow_real=True`, and a failing cell becomes an error row rather than a
dead sweep.
