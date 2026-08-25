# PIPELINE — running the pre-registered evaluation

How results tables are produced once the validation gates close. The runner is
`src/mortcal/runner.py`; the pre-registered regimes are `src/mortcal/splits.py`
(a transcription of PREREGISTRATION.md — editing either is a reportable
protocol deviation).

## Gate status (2026-08-25)

| Gate (PREREGISTRATION.md) | Status |
|---|---|
| 1. Synthetic truth (`tests/test_synthetic_calibration.py`) | **PASSES** — correctly-specified Poisson-LC attains nominal coverage; the harness detects an injected break as under-coverage. Known documented finding: two-stage SVD-LC is mildly overdispersed in-DGP. |
| 2. Oracle parity (Python LC/PLC vs R StMoMo, 1e-6 relative on α, β, κ) | **OPEN** — R 4.6.1 installed; StMoMo not yet installed; parity script not yet run. |
| 3. Life-table parity (our ex vs HMD published ex) | Spot-check harness in `tests/test_lifetable.py`; full HMD spot checks run with the first real-data pass. |
| Reading gate: Dowd et al. (2010b), Schnürch & Korn (2022) before related-work text | **OPEN** — SSRN links in `literature/GET-THESE.md`, browser-only. |

**The runner enforces gate 2 in code**: `run_regime` raises unless the regime
is named `"synthetic"` or `allow_real=True` is passed. Do not pass
`allow_real=True` until parity is demonstrated and recorded.

## What the runner does

- `MODELS` — classical families in scope for the CPU sweep: `LC`
  (Lee–Carter SVD), `PLC` (Poisson Lee–Carter), `CBD` (M5), `RH` (M2-A),
  `SVAR` (banded VAR on improvements).
- `MECHANISMS` — `native`, `pboot` (Poisson bootstrap), `split_conf`,
  `enbpi`, `copula_conf`. Admissibility is transcribed from `docs/GRID.md`;
  inadmissible or unknown cells raise `InadmissibleCellError` before any fit.
- `run_cell(D, E, model, mechanism, h, n_samples, rng, obs_D=…, obs_E=…)`
  fits one grid cell on the training matrices `[n_ages, n_train_years]`,
  draws `[n_samples, h, n_ages]` predictive m_x paths, and computes through
  the single evaluation path (`mortcal.eval` + `mortcal.lifetable`):
  RMSE/MAE/CRPS on log m_x (point forecast = pointwise median of log
  samples), Poisson log score on observed deaths (rounded-deaths convention,
  PREREGISTRATION.md "Metrics"), coverage and Winkler at 50/80/95, joint
  path coverage at 95, PIT KS statistic, and the H5 inputs — mean / 2.5% /
  97.5% quantiles plus observed values for e0, e65 and ä65 @2% from the
  horizon-1 sample table.
- `run_regime(panel_df, regime, models, mechanisms, n_samples, out_path)`
  sweeps (pop × sex × origin × model × mechanism) and writes ONE parquet row
  per cell (pandas + pyarrow). A cell whose fit or scoring raises is skipped
  and logged; the exception string lands in the row's `error` column so the
  sweep always finishes and failures are auditable in the output itself.
  Per-cell seeds derive from the global seed 20260825 via
  `SeedSequence([base, origin, crc32(pop), crc32(sex), crc32(model),
  crc32(mech)])` and are recorded in the `seed_entropy` column (rule 7).

### Conformal cells and proper scores

The conformal wrappers emit uniform-in-interval samples (see the module
docstring of `src/mortcal/uq/conformal.py`): intervals, not distributions.
Every row carries a boolean `scores_secondary` column — `True` for
`split_conf` / `enbpi` / `copula_conf` — and CRPS / log score / PIT from
flagged rows go only to a flagged appendix table, never ranked against
distributional mechanisms.

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
           allow_real=True)   # ONLY after gate 2 closes — see guard

# Stable (control): 13 expanding origins, one Regime each
run_regime(panel, REGIMES["stable"], list(MODELS), list(MECHANISMS),
           n_samples=2000, out_path="results/stable.parquet",
           allow_real=True)

# Placebo break: train <= 1913, test 1914-1922, 11 populations
run_regime(panel, REGIMES["placebo"], list(MODELS), list(MECHANISMS),
           n_samples=2000, out_path="results/placebo.parquet",
           allow_real=True)
```

Note `pboot` currently requires the base class to implement `fitted_mx()`;
`CBD`, `RH` and `SVAR` do not yet, so those cells raise a clear
`NotImplementedError` (recorded as error rows if swept). Implement
`fitted_mx()` on those families before the full sweep.

## Deliberately NOT implemented yet

- **Neural families** (neural-LC, CNN-LC, LSTM-on-k_t, distributional NB
  head) and the **multi-output GP** — GPU-budget arms of the grid; the
  runner registries hold classical families only.
- **Deep ensemble / MC dropout mechanisms** — only meaningful for the neural
  arms (inadmissible for classical fits per `docs/GRID.md`), so absent from
  `MECHANISMS` entirely.
- **Inference layer** — wild cluster bootstrap over populations
  (20 correlated clusters), Diebold–Mariano, and the 90% Model Confidence
  Set operate on the parquet outputs downstream; not in the runner.
- **Analysis/figures** — calibration-by-age curves, Murphy decomposition and
  H1–H5 tests consume the per-cell parquet rows; separate analysis stage.
- **STMF weekly robustness, 2024-provisional and age-cap sensitivities** —
  planned secondary runs (PREREGISTRATION.md "Planned robustness").

## Tests

`tests/test_runner.py` exercises the sweep end to end on the synthetic
Poisson-LC truth (2 populations, 2 origins, LC+PLC × native+split_conf),
asserts every metric finite, the parquet round-trips, real regimes are
refused while gate 2 is open, and a failing cell becomes an error row rather
than a dead sweep.
