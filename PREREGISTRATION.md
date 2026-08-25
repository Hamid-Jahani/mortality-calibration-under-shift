# Pre-registration — Mortality interval calibration across the COVID-19 break

**Registered:** 2026-08-25, before any model has been fit to real HMD data.
**Data vintage:** HMD bulk files, last modified 2026-06-15, Methods Protocol v6,
pinned by `data/MANIFEST.sha256`. **Seed:** global 20260825; per-ensemble-member
seeds = global + member index, recorded in results metadata.

This document fixes the design. Deviations will be reported as deviations.

## Question

Do the prediction intervals of stochastic, machine-learning and neural mortality
forecasting models attain their nominal coverage when the evaluation window
crosses a structural break (COVID-19), and does the answer depend on the model
family, the uncertainty mechanism, or both?

## Hypotheses (fixed before estimation)

- **H1** Model rankings by RMSE and by CRPS disagree (rank correlation < 1;
  at least one top-3 inversion) in the shift regime.
- **H2** Nominal 95% intervals under-cover in the shift regime for every
  model family; the shortfall exceeds the stable-regime shortfall.
- **H3** Joint path coverage over h = 1…5 is materially below marginal
  coverage at the same nominal level for every family (reported gap; test:
  paired difference > 0).
- **H4** Coverage failure is non-uniform in age: shift-regime coverage at ages
  65–99 is lower than at ages 25–64 for every family.
- **H5** Interval miscalibration propagates to derived quantities: empirical
  coverage of 95% intervals for e65 and annuity factor ä65 (2%) departs from
  nominal by more than the corresponding stable-regime departure.

No directional claim is registered about neural-vs-classical ordering; the
crossed design estimates it.

## Data

- HMD 1×1 deaths and central exposures, ages 0–99 (cells with E=0 are all at
  ages ≥90 in this vintage; the 110+ open group and model-smoothed old ages are
  excluded), sexes modelled separately.
- Poisson likelihood with log-exposure offset throughout; rates never modelled
  as Gaussian on the raw scale.

### Regimes and populations (fixed now)

**Shift (primary).** Train ≤2019, test 2020–2024, horizons h=1…5.
The 20 populations whose final annual data reach 2024 in this vintage:
BEL, CHE, CHL, DNK, EST, FIN, HKG, HRV, ISL, JPN, KOR, LTU, LUX, LVA, NOR,
PRT, SVK, SWE, TWN, USA.

**Stable (control).** Expanding origin: train ≤T for T = 1990, 1992, …, 2014;
test T+1…T+5 (all test years ≤2019). Same 20 populations, subject to each
population's own start year.

**Placebo break.** Train ≤1913, test 1914–1922, h=1…9. Populations with
continuous 1908–1922 coverage and start ≤1900, excluding Belgium (missing
occupation-era rows 1914–1918): CHE, DNK, FIN, FRATNP, GBRTENW, GBR_SCO, ISL,
ITA, NLD, NOR, SWE. FRACNP and GBRCENW are excluded as overlapping variants of
FRATNP and GBRTENW.

## Factors

**Model families (10):** Lee–Carter (SVD+RWD); Poisson Lee–Carter (Brouhns);
CBD (M5); APC / Renshaw–Haberman (M2-A); sparse VAR on log-rates; multi-output
GP; neural-LC (Richman–Wüthrich embeddings); shallow CNN-LC (Perla-style);
LSTM-on-k_t; distributional Poisson/NB-head network.

**UQ mechanisms (7):** model-native; semiparametric Poisson bootstrap;
deep ensemble (M=10); MC dropout; split conformal; EnbPI; copula/joint
conformal. Not every cell is admissible (e.g. MC dropout on Lee–Carter);
the admissible sub-grid is enumerated in `docs/GRID.md` (50 primary cells) and claims
are stated as contrasts within admissible sub-grids, never as full-factorial
main effects.

Conformal calibration sets are drawn only from data available at the training
cutoff (inner time splits). No mechanism sees test years.

## Metrics — all computed from predictive samples through one code path

Point: RMSE and MAE on log m̂x; e0 error.
Probabilistic: CRPS (sample estimator) on log mx; Poisson log score on observed
deaths **rounded to the nearest integer** (HMD deaths are Lexis-split and
fractional; this convention is fixed now; sensitivity: CRPS on death counts);
Winkler/interval score and empirical coverage at 50/80/95%; randomized PIT with
uniformity test; calibration-by-age curves; joint path coverage over h (per
population-sex-age trajectory); Murphy calibration–resolution decomposition.
Actuarial: interval coverage and error for e65, e0, and ä65 at 2% interest
computed from each predictive sample's period life table.
Inference: Diebold–Mariano per population-sex on CRPS with wild cluster
bootstrap over populations; Model Confidence Set at 90%.

## Validation gates (must pass before real-data results are read)

1. **Synthetic truth:** data simulated from a known Poisson Lee–Carter process;
   the correctly-specified model must attain 95% nominal coverage within
   Monte-Carlo tolerance at h=1…10, and PIT must pass uniformity. Gate is a CI
   test (`tests/test_synthetic_calibration.py`).
2. **Oracle parity:** Python Lee–Carter and Poisson-LC parameters must match
   R StMoMo on one shared subset (α, β, κ within 1e-6 relative) before their
   forecasts are used.
3. **Life-table parity:** our ex computed from HMD mx must reproduce HMD's
   published ex within publication rounding for spot checks.

## Splits discipline

Expanding-origin time splits only; hyperparameters tuned on inner time splits
(final 5 pre-cutoff years of each training window); the test years are touched
exactly once, by the final scoring run.

## Planned robustness (secondary, not gating)

STMF weekly data for break timing; sensitivity to 2024-provisional revisions
(re-run shift regime dropping 2024); sensitivity to age cap 90 vs 99.
