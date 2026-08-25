# Session state — 2026-08-26

## Where things stand

62 → **88 tests green**. Committed through: prereg (hash-stamped), all five
classical models, UQ wrappers (bootstrap + 3 conformal), scoring suite,
life-table/annuity chain, splits, experiment runner, synthetic smoke run.
Runner defect ledger 1–6 CLOSED 2026-08-26 (below); `docs/PIPELINE.md` has
the column glossary.

**Gates:** 1 (synthetic truth) ✅ · 2 (StMoMo oracle parity) ✅ machine-precision
· 3 (life-table parity vs HMD published ex) ✅ in test suite · literature gate
(Dowd 2010b, Schnürch–Korn 2022) ❌ **user's two SSRN browser downloads** ·
runner metric gaps ✅ closed.

## Runner defects (adversarial verifier) — ALL CLOSED 2026-08-26

1. ✅ **H4 calibration-by-age** — `coverage_{50,80,95}_band{0_24,25_64,65_99}`,
   `pit_ks_band*`, plus `cov95_by_age` (JSON list, one entry per panel age,
   null where masked) so age curves plot from parquet without refits.
2. ✅ **Murphy decomposition** — `mortcal.eval.murphy_decomposition` /
   `murphy_pit`; runner emits `murphy_{reliability,resolution,uncertainty,brier}`
   on the 95% hit indicators (constant-forecast form: reliability = squared
   coverage gap) and `murphy_pit_*` + `pit_hist` from the PIT values.
3. ✅ **CRPS on counts** — `crps_counts` against UNROUNDED observed deaths;
   PLC/RH via `sample_deaths`, everything else composes Poisson(E·sample_mx)
   (the same construction) on masked ages; rounded-deaths log score kept.
4. ✅ **Per-horizon series** — `crps_h{k}`, `logscore_h{k}`, `coverage95_h{k}`,
   `winkler95_h{k}` for k=1..H per (pop, sex, origin); scalar means kept.
5. ✅ **Minor trio** — `round_deaths` = floor(d+0.5) half-up; ONE point
   functional (pointwise median of samples) for RMSE/MAE and `{e0,e65,ann65}_point`
   with explicit `*_error = point − obs` (sample-mean columns dropped);
   `fitted_mx()` on CBD/RH/SVAR → pboot builds for all five families, the
   `NotImplementedError` path is gone.
6. ✅ **CBD age restriction** — `MODEL_KWARGS = {"CBD": {"age_min": 55}}` reaches
   the family under every mechanism; runner masks NaN ages, records
   `n_ages_scored` / `n_cells`, and e65/ä65 come exactly from the truncated
   table (ratio invariance, tested). Known residual: CBD × conformal loses ages
   55–64 too, because `_conformal_quantile` pools NaN residuals inside the
   25–64 Mondrian band (scores 65–99 only; `n_ages_scored` = 35). Fix belongs in
   `mortcal/uq/conformal.py` (nan-aware band quantile), not the runner.

## NEW — needs a design decision BEFORE any real-data run (found 2026-08-26)

**Scoring target for the rate-scale metrics is unspecified and it matters.**
`run_cell` scores `sample_mx` (the LATENT m_x, no Poisson noise) against the
OBSERVED crude rate D/E, which carries Poisson noise. Gate 1 passes because
it scores against the TRUE latent m_x — a synthetic-only luxury. Synthetic
smoke (PLC native, its own DGP, h=5): coverage_95 = 0.96 vs latent truth,
0.61 (40 ages) / 0.20 (100 ages) vs observed crude, 0.94–0.96 when Poisson
noise is composed into the rate samples (D̃ ~ Poisson(E·m), score log(D̃/E)).
Consequences if left as is: (a) a correctly-specified model fails nominal
coverage on the real-data code path (rule 5 violated in spirit); (b) the
"coverage failure concentrates at young ages" H4 signal would be partly
sampling noise (few deaths → noisy crude rates), worst for ISL/LUX;
(c) SVAR (fit on observed improvements, so its innovation variance absorbs the
noise) and the conformal wrappers (calibrated on observed residuals) look
better than PLC/LC/RH for a reason unrelated to shift — smoke: SVAR native
0.958, conformal ~0.92, PLC native 0.16. Options: score the OBSERVABLE's
predictive law (compose Poisson noise into rate samples for CRPS/coverage/PIT
— the count-scale log score already does this) as the primary, latent as a
sensitivity; or keep latent and report noise-floor bands. Pre-registration
("CRPS on log mx", "empirical coverage") does not fix this; decide, record as
a clarification, then re-run gate 1 THROUGH `run_cell` with observed truth.

## Then

7. Neural families (torch via uv, GPU): neural-LC embeddings, CNN-LC, LSTM-kt,
   NB head + deep-ensemble/MC-dropout mechanisms.
8. Multi-output GP (gpytorch) or document exclusion.
9. Wild-cluster-bootstrap inference layer + MCS.
10. STABLE regime real-data run (after 1–5) — first real numbers.
11. Placebo prerequisite: WWI country-notes check; 2024-final-vs-revisable check.

## Don'ts

- Do NOT resume workflow `wf_75bb90b5-839` (dead; resume clobbers manual RH
  gauge fix + conformal tests).
- Do NOT commit `results/parity/D_*.csv` / `E_*.csv` (raw HMD, gitignored).
- Real-data runs stay behind `allow_real=True` (guard kept as the auditable
  act even though defects 1–6 and gate 2 have closed).

## Key numbers to remember

- Parity: β rel diff 2.4e-14, identical loglik (StMoMo 0.4.1, SWE ♂ 1950–2000).
- RH bug story: degree-2 cohort constraint documented-but-not-implemented by
  the killed agent → κ bent by 3.7 units → coverage 0.68; fixed → nominal.
- Conformal repair test: native 0.72 → wrappers ≥0.85 marginal; joint 0.40 → 0.75+.
