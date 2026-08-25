# Probabilistic Mortality Forecasting under Distribution Shift

Calibration, Uncertainty, and Neural Extensions of Lee–Carter.

A reliability audit of probabilistic mortality forecasts across a structural
break. The contribution is the audit and the reusable evaluation protocol — not
a new architecture.

**Status:** pre-modelling. No model has been fit. `PREREGISTRATION.md` is not yet
written, and by the rules below nothing gets fit until it is committed.

## Research question

Neural extensions of Lee–Carter that emit prediction intervals already exist.
What has not been established is whether those intervals hold their nominal
coverage when the mortality surface breaks. This repository tests that.

Five pre-registered hypotheses (H1–H5) covering score-vs-RMSE ranking
disagreement, marginal under-coverage, joint path coverage, age-localised
failure, and propagation into annuity and life-expectancy quantities.

## Data

**Human Mortality Database** — 49 populations, vintage 15 Jun 2026.

The `Dataset/` tree is **not committed**. HMD's user agreement restricts
redistribution: obtain the data yourself from
<https://www.mortality.org/Data/ZippedDataFiles> after free registration.

`data/MANIFEST.sha256` pins the exact file vintage this work uses. Verify a
local copy from the repository root:

```
sha256sum -c data/MANIFEST.sha256
```

## Methodology rules

These are binding, not aspirational.

1. `PREREGISTRATION.md` is committed and hash-stamped **before** the first model fit.
2. Expanding-origin temporal splits only. Never a random split — mortality panels leak trivially.
3. Hyperparameters tune on an inner *time* split, never on test.
4. Every model emits a predictive distribution, not a point forecast.
5. The evaluation harness is validated against synthetic truth from a known
   Lee–Carter data-generating process before it touches real data.
6. Python Lee–Carter matches R StMoMo to numerical tolerance on α, β, k.
7. Seeds recorded per ensemble member.

## Stack

Python (neural models, evaluation) + R (StMoMo/demography classical baselines
and numerical oracle).

## Layout

```
Dataset/       HMD files — untracked, see data/MANIFEST.sha256
data/          manifest and data documentation
literature/    curated notes (INDEX, MISSING, GAP-ANALYSIS); PDFs untracked
```
