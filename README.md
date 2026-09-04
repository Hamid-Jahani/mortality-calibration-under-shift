# Do Mortality Prediction Intervals Survive a Pandemic?

A Pre-Registered Calibration Audit of Ten Forecasting Families and Seven
Uncertainty Mechanisms Across the COVID-19 Break.

A reliability audit of probabilistic mortality forecasts across a structural
break. The contribution is the audit and the reusable evaluation protocol — not
a new architecture. Model family and uncertainty mechanism are **crossed
factors** (50 admissible cells, `docs/GRID.md`), so coverage failure can be
attributed to the architecture or to the uncertainty machinery bolted onto it.
Title and design selected by an adversarial multi-agent tournament —
see `docs/IDEA.md` for the full record.

**Status:** complete. All three pre-registered regimes are swept, QA-gated
(zero machine-failure rows) and analysed; the paper draft is written from
them. `PREREGISTRATION.md` was committed 2026-08-25 — sha256
`bbbe3860a5446d063e186e56555afa893b07c36cf15a990d07567471affe50be` — before
the first model was fitted to real data, and every deviation since is
recorded in the four addenda and `docs/STATUS.md`.

Headline findings, all from the pre-registered grid:

- **Most miscalibration predates the pandemic.** In the quiet stable control
  (13 expanding origins, 1990–2014), classical native intervals already cover
  0.702 (Lee–Carter), 0.682 (Poisson-LC) and 0.731 (APC/RH) against a nominal
  0.95.
- **The registered one-sided H2 is refuted.** The two-sided
  |coverage − nominal| grows from stable to shift in only 27 of 50 cells.
  Coverage falls in 34 of 50, but that is degradation only for arms that began
  below nominal: over-covering arms move *toward* nominal. The break sorts
  families by which side of nominal they started on.
- **H3 holds universally** — joint path coverage is below marginal in 100 of
  100 (regime, cell) pairs.
- **H5 is the practical warning.** Annuity-factor intervals are far worse than
  rate-level ones and are already broken in quiet periods: Lee–Carter's
  nominal 95% interval for ä₆₅ covers 0.253 of realisations in the stable
  control.

Validation gates all pass: synthetic truth, R/StMoMo oracle parity
(β relative difference 2.4e-14), and life-table parity.

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
docs/          IDEA.md (tournament record) · GRID.md (admissible cells)
literature/    curated notes (INDEX, TRIAGE, GAP-ANALYSIS, GET-THESE); PDFs untracked
src/mortcal/   data/ (HMD parser) · models/ (LC, PLC, CBD, RH, sVAR)
               uq/ (bootstrap, conformal) · eval/ (scores) · lifetable · splits
tests/         50 tests; synthetic-truth calibration gates included
```

Run the suite: `uv run pytest -q` (Python pinned 3.12 via uv).

## License

Released under the [MIT License](LICENSE).
