# Session state — 2026-08-26 (afternoon)

## Where things stand

88 → **99 tests green**. Committed through: prereg + addenda 1 and 2
(hash-stamped), all five classical models, UQ wrappers (bootstrap + 3
conformal), scoring suite, life-table/annuity chain, splits, experiment
runner, synthetic smoke run. Runner defect ledger 1–6 CLOSED 2026-08-26.
**Addendum 2 is now implemented and test-guarded** (was registered but not
implemented — see below). `docs/PIPELINE.md` has the column glossary.

**Gates:** 1 (synthetic truth) ✅ · 2 (StMoMo oracle parity) ✅
machine-precision · 3 (life-table parity vs HMD published ex) ✅ in test
suite · **literature gate ✅ CLOSED** — all nine manual-fetch papers obtained
2026-08-26, 59 PDFs / 52 extracted texts, see `literature/GET-THESE.md` ·
runner metric gaps ✅ closed · **addendum-2 implementation ✅ closed**.

## Addendum 2 — CLOSED 2026-08-26

`PREREGISTRATION-ADDENDUM-2.md` was committed at 02:06 but the code still did
the opposite. All four clauses now implemented in `mortcal/runner.py` and
`mortcal/eval/scores.py`, guarded by `tests/test_addendum2_scoring.py` (11
tests).

1. ✅ **Rate-scale scores are Poisson-inclusive.** `run_cell` composes
   D* ~ Poisson(E · m_x*) on the model's own paths and scores log crude
   rates. Measured before the fix, correctly-specified PLC on its own DGP:
   coverage **0.104 / 0.189 / 0.278** against nominal 0.50 / 0.80 / 0.95.
   After: nominal within tolerance at all three levels. This is validation
   gate 1 now enforced *through the real-data code path*, not only against a
   latent truth available in simulation.
2. ✅ **Half-count continuity correction.** `mortcal.eval.scores.log_crude_rate`
   = log(max(D, 0.5)/E), applied to BOTH sides. Replaces the 1e-10 rate floor,
   which sent a zero-death cell to log(1e-10) = −23.03 instead of ≈ −8.3 —
   finite, so the mask kept it, at ~85× normal squared error. Measured: one
   injected zero-death cell in 500 inflated `rmse_logmx` **3.8×** (0.124 →
   0.475). In the 2020–2024 window that is 10.4% of cells for ISL and LUX
   (52 of 500 each). New column `n_zero_death_cells`.
3. ✅ **Conformal scored at construction level only.** `coverage_{50,80}`,
   `winkler_{50,80}` and `coverage_{50,80}_band*` are NaN on conformal rows.
   Uniform-in-interval samples put exactly 50% of mass in the middle half, so
   `coverage_50` trended to nominal by construction and flattered these arms.
4. ✅ **PIT p-value.** New column `pit_ks_pvalue`, descriptive only (PIT values
   are dependent across ages/horizons; formal inference uses the clustered
   procedures in `mortcal.inference`).

Known residual, unchanged: CBD × conformal loses ages 55–64 because
`_conformal_quantile` pools NaN residuals inside the 25–64 Mondrian band.
Fix belongs in `mortcal/uq/conformal.py` (nan-aware band quantile).

## NEW BLOCKER — needs a registered decision before the shift sweep

**Five of twenty shift populations cannot be fit on the full expanding
window.** `_pivot_matrices` refuses non-finite training cells (correctly — it
will not silently model holes). Audit of all 20 SHIFT populations × 2 sexes,
training years ≤ 2019, ages 0–99:

| pop | sex | NaN cells | affected ages | first clean year |
|---|---|---|---|---|
| CHE | male | 10 | 98, 99 | 1917 |
| CHE | female | 1 | 99 | 1887 |
| FIN | male | 27 | 97–99 | 1947 |
| ISL | male | 30 | 96–99 | 1953 |
| ISL | female | 13 | 95–99 | 1924 |
| LUX | male | 6 | 97–99 | 1969 |
| SWE | male | 12 | 98, 99 | 1874 |

8 of 40 (pop, sex) cells affected; the other 15 populations are clean.
`run_regime` records these as `error` rows rather than crashing, so the
failure mode is **silent loss of 5 populations including SWE** — the flagship
long series — plus ISL and LUX, the small-population stress cases.

No age cap resolves it cleanly: remaining NaN training cells are 99 at
age_max=99, 29 at 98, **8 at 97**, and still 1 at 95 (ISL female). Options,
all requiring registration as a clarification:

- **(a) age cap.** Simple, but the surviving cap (≈94) removes exactly the
  ages where H4 predicts coverage failure concentrates. Weakens the paper.
- **(b) per-(pop, sex) training start** at the first clean year. Preserves the
  age range; makes the expanding window population-dependent, which must be
  reported and may affect cross-population comparability.
- **(c) NaN-aware fitting** — mask individual cells inside the likelihood.
  Most faithful, most work; every family needs it.
- **(d) accept the loss** and report 15 populations. Cheapest, but dropping
  SWE from a mortality paper needs a strong justification.

Recommendation is (b) or (c); this has NOT been decided and no sweep should
run until it is.

## Then

5. Neural families (torch via uv, GPU): neural-LC embeddings, CNN-LC, LSTM-kt,
   NB head + deep-ensemble/MC-dropout mechanisms.
6. Multi-output GP (gpytorch) or document exclusion.
7. Wild-cluster-bootstrap inference layer + MCS.
8. STABLE regime real-data run (after the blocker above) — first real numbers.
9. Placebo prerequisite: WWI country-notes check; 2024-final-vs-revisable check.
10. Related-work text: Dowd 2010b and Schnürch–Korn 2022 are now on disk and
    must be READ — the lexical scan in `literature/GET-THESE.md` shows both
    evaluate interval performance (Schnürch–Korn via PICP, Dowd via
    exceedances), so **H2 is replication, not discovery**, and the novelty
    rests on H3 / the crossed design / proper scores / conformal arms.

## Environment — BROKEN, workaround in use

`.venv` was built under Windows profile `C:\Users\Gaming`, which no longer
exists (only `C:\Users\ASUS`). `uv` is not on PATH for this profile either, so
CLAUDE.md's `~/.local/bin/uv.exe` is stale. Patching `.venv/pyvenv.cfg` is NOT
enough — the uv trampoline has the path compiled in. All 86 site-packages are
intact. Same root cause as the git `dubious ownership` error (fixed with
`safe.directory`).

Run the suite with:

```
PYTHONPATH="<repo>/.venv/Lib/site-packages;<repo>/src" \
  "C:/Users/ASUS/AppData/Roaming/uv/python/cpython-3.12.13-windows-x86_64-none/python.exe" \
  -m pytest -q
```

Durable fix: reinstall `uv` for the ASUS profile and recreate `.venv`.

## Don'ts

- Do NOT resume workflow `wf_75bb90b5-839` (dead; resume clobbers manual RH
  gauge fix + conformal tests).
- Do NOT commit `results/parity/D_*.csv` / `E_*.csv` (raw HMD, gitignored).
- Real-data runs stay behind `allow_real=True`.
- Do NOT run the shift sweep until the missing-cell decision above is
  registered.

## Key numbers to remember

- Parity: β rel diff 2.4e-14, identical loglik (StMoMo 0.4.1, SWE ♂ 1950–2000).
- Pre-addendum-2 coverage of a correctly-specified model: 0.104 / 0.189 / 0.278.
- Zero-death floor artefact: one cell in 500 → RMSE 0.124 → 0.475 (3.8×).
- ISL and LUX carry 52 zero-death cells each in the 2020–2024 window (10.4%).
- RH bug story: degree-2 cohort constraint documented-but-not-implemented by
  the killed agent → κ bent by 3.7 units → coverage 0.68; fixed → nominal.
- Conformal repair test: native 0.72 → wrappers ≥0.85 marginal; joint 0.40 → 0.75+.
