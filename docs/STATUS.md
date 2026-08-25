# Session state — 2026-08-25 end of day

## Where things stand

54 → **62 tests green**. Committed through: prereg (hash-stamped), all five
classical models, UQ wrappers (bootstrap + 3 conformal), scoring suite,
life-table/annuity chain, splits, experiment runner, synthetic smoke run.

**Gates:** 1 (synthetic truth) ✅ · 2 (StMoMo oracle parity) ✅ machine-precision
· 3 (life-table parity vs HMD published ex) ✅ in test suite · literature gate
(Dowd 2010b, Schnürch–Korn 2022) ❌ **user's two SSRN browser downloads** ·
runner metric gaps ❌ below.

## FIRST TASKS TOMORROW — runner defects (adversarial verifier, all confirmed)

1. **H4 blocked:** `run_cell` emits only scalar mean coverage — no
   calibration-by-age. Add per-age-band columns (0–24, 25–64, 65–99) for
   coverage_50/80/95 + PIT, or persist per-cell indicators.
2. **Murphy decomposition missing** — pre-registered metric, no implementation
   in `mortcal/eval`. Implement calibration–resolution split of the Brier/quantile
   score family.
3. **CRPS-on-counts sensitivity missing** — registered companion to the
   rounded-deaths log-score convention. Add `crps_counts` (needs `sample_deaths`
   paths and obs D).
4. **Per-horizon granularity destroyed** — DM test needs loss differentials;
   store `crps_h1..h5` columns, not just the mean.
5. Minor: `np.round` is banker's rounding — switch to `np.floor(d+0.5)` or
   document; e0 uses sample-mean while log-mx uses median — unify point
   functionals; `fitted_mx()` missing on CBD/RH/SVAR so their pboot cells error.

## Then

6. Wire CBD age-restriction (55–99) into runner registry kwargs.
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
- Real-data runs stay behind `allow_real=True` until defects 1–5 close.

## Key numbers to remember

- Parity: β rel diff 2.4e-14, identical loglik (StMoMo 0.4.1, SWE ♂ 1950–2000).
- RH bug story: degree-2 cohort constraint documented-but-not-implemented by
  the killed agent → κ bent by 3.7 units → coverage 0.68; fixed → nominal.
- Conformal repair test: native 0.72 → wrappers ≥0.85 marginal; joint 0.40 → 0.75+.
