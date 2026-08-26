# Session state — 2026-08-26 (evening)

## Where things stand

**119 tests green.** Registered documents: PREREGISTRATION.md + addenda 1–3,
`docs/PRIOR-ART.md` — all hash-stamped in their commits. Addenda 2 AND 3 are
**implemented and test-guarded** (`tests/test_addendum2_scoring.py`,
`tests/test_addendum3.py` — the latter uses low-exposure fractional-death
worlds because the E=1e5 worlds of `test_runner.py` have no mutation power).

**Gates:** 1 (synthetic truth, now through the real-data code path) ✅ ·
2 (StMoMo oracle parity) ✅ **re-verified after the §1 weighted-fit rewrite**:
α 3.1e-15, β 2.4e-14, κ 1.5e-13 · 3 (life-table parity) ✅ in suite ·
literature gate ✅ CLOSED (all nine papers on disk; verdicts in
`docs/PRIOR-ART.md`) · runner ledgers 1–6 ✅ · addendum-2 ledger B1–B7 ✅
(see below) · **missing-cell blocker ✅ RESOLVED** (addendum 3 §1 — option (c),
chosen 3–0 by independent judge lenses; the 99 "missing" cells are verified
structural zeros with zero Fisher information, already anticipated at
PREREGISTRATION.md:37).

## What was implemented for addendum 3 (2026-08-26 evening)

- **§1** `W = 1{E>0}`: PLC (naturally-weighted Newton + NaN-safe half-count
  init), LC (EM-SVD over missing cells, exact SVD fast path), RH (exposure
  weights folded into its cohort-clip W), CBD (weighted per-year 2-param
  solve, closed-form fast path), SVAR (common-complete subsample; measured
  worst row loss FIN·m 140→101, all defective years pre-1970).
  `hmd.build_panel` keeps E=0 rows (mx NaN); `PoissonBootstrap` guards
  0·NaN in D_hat.
- **§2/§4** `_pivot_matrices` trims to the maximal contiguous year block
  ending at the origin (BEL → trains 1919–2019) and enforces n_train ≥ 15.
- **§3** test-window E=0 ages masked; derived quantities on the maximal
  contiguous scored block, range reported (`derived_age_lo/hi`).
- **§5** rate-scale truth = `log(max(round_deaths(D), 0.5)/E)` — observed
  side joins the integer lattice of the Poisson-inclusive samples.
- **§6** conformal 95% coverage/Winkler/joint from the wrapper's own
  `interval(h)` bounds at its own α — no sample quantiles, no Poisson
  composition; calibration residuals on the half-count scale; NaN-aware
  `_conformal_quantile`.
- **§7** SVAR rejects explosive coefficient draws (companion spectral
  radius ≥ 1, redraw cap 100); rates never clipped.
- **§9** conformal `cal_years`/`h_cal` 8 → 9 (placebo h=9 now calibrated).
- **§10** `n_zero_death_cells`: pre-mask, full panel, `D < 0.5`.
- B1: the 1e-10 rate floor is gone from every model ingest
  (lc/rh/svar/cbd/conformal); the ONE remaining floor is the observed life
  table (`runner.py` `obs_full`), which is correct and documented in §5 of
  the addendum (half-count there would shift ISL·f 2020 e0 by −0.370 y,
  beyond gate 3's 0.15 y tolerance).

Measured stakes, for the record: pre-fix, a correctly-specified PLC scored
0.104/0.189/0.278 against nominal 0.50/0.80/0.95; the conformal H3 gap would
have published as 3.8pp instead of 22pp (6× understatement); BEL was silently
splicing 1913→1919; one zero-death cell in 500 inflated RMSE 3.8×.

## Novelty position (registered — docs/PRIOR-ART.md + addendum 3 §8)

H1 demoted to harness-validity check (Barigou 2021, Goes 2024, Schnürch–Korn
Table 2 all publish the inversion). H2 two-sided (stable-regime universality
already falsified: FFNN 97.0/CNN 98.0 PICP; LC-LSTM PICP=1.0); its SHIFT half
is the open question — the field declined the out-of-sample pandemic test for
lack of post-break data (Goes et al., quoted). H3 compared to model-implied
joint coverage (registered form was near-tautological). H4 reversal
pre-committed as informative (Dowd: performance improves with age). H5
restricted to ex-post derived-quantity coverage; attenuation named as the
alternative. The paper's spine: the crossed family × mechanism design
(Schnürch–Korn state verbatim they held the mechanism fixed) + conformal arms
(0 corpus hits in mortality) + H3's measurement + H5's audit.

## DECIDED 2026-08-26 — build the missing five families

**User decision: keep the tournament-locked title and build the family axis**
(multi-output GP, neural-LC Richman–Wüthrich, shallow CNN-LC Perla, LSTM-on-
k_t, distributional NB head) plus the deep-ensemble (M=10) and MC-dropout
mechanisms that exist only on those rows. The classical sweep WAITS and runs
once, as production, when the grid is complete (~11 min on 8 cores; compute
is not the constraint). Sequence: environment repair (uv + .venv with torch,
pinned 3.12) → written spec per family (hyperparameters tune on inner TIME
splits; every family emits sample paths through the single interface; per-
family validation gates on synthetic truth) → implement → gates → production
sweeps.

## Then

1. Neural families (torch via uv, GPU): neural-LC embeddings, CNN-LC,
   LSTM-k_t, NB head + deep-ensemble/MC-dropout mechanisms.
2. Multi-output GP (gpytorch) or document exclusion.
3. Inference layer wiring: `mortcal.inference` has ZERO call sites and
   **silently inverts under NaN** (one hole flips the MCS to eliminate the
   good models at p=0.000; `dm_wild_cluster` returns t=inf, p=0.0). Needs a
   runner-row → [n_units, n_models] adapter (does not exist) and a
   raise-on-NaN guard. Highest-priority LATER item.
4. crps_counts shared draw (currently an independent redraw — MC noise +
   cross-family asymmetry).
5. Sweeps: STABLE first, then SHIFT, then PLACEBO (`scripts/run_regime.py`,
   still passes `allow_real=True` explicitly). Report per-origin effective
   cluster counts (addendum 3 §4).
6. Two fetches before write-up: Schnürch–Korn Online Supplement C.3
   (annuity PVs — the only live threat to H5) and Stankevičiūtė 2021 CF-RNN
   (baseline for the copula-conformal arm).
7. Placebo prerequisite closed: test-window E=0 cells handled by §3;
   2024-final question answered in docs/DATA-PREREQS.md §B (register S-S1/
   S-S2/S-S3 sensitivities at analysis time).

## Environment — BROKEN, workaround in use

`.venv` was built under Windows profile `C:\Users\Gaming`, which no longer
exists. `uv` is not on PATH. Patching `pyvenv.cfg` is NOT enough (path baked
into the trampoline exe). All 86 site-packages intact. Run tests with:

```
PYTHONPATH="<repo>/.venv/Lib/site-packages;<repo>/src" \
  "C:/Users/ASUS/AppData/Roaming/uv/python/cpython-3.12.13-windows-x86_64-none/python.exe" \
  -m pytest -q
```

Durable fix: reinstall uv for the ASUS profile, recreate .venv.

## Don'ts

- Do NOT resume workflow `wf_75bb90b5-839` (dead; resume clobbers manual
  fixes). The decision workflow `wf_9a3ba779-871` is complete — its per-agent
  evidence lives in its journal if a claim needs re-checking.
- Do NOT commit `results/parity/D_*.csv` / `E_*.csv` (raw HMD, gitignored).
- Real-data runs stay behind `allow_real=True`.
- Do NOT clip SVAR rates/λ to rescue divergent draws (addendum 3 §7 —
  rejection only).

## Key numbers to remember

- Parity after weighted rewrite: α 3.1e-15, β 2.4e-14, κ 1.5e-13.
- Pre-addendum-2 coverage of a correctly-specified model: 0.104/0.189/0.278.
- Conformal sample-quantile defect: marginal/joint 0.995/0.957 vs
  0.960/0.740 on true bounds.
- Zero-death cells (sex-specific, 2020–24): ISL·f 117 (23.4%), LUX·f 110
  (22.0%), ISL·m 68, LUX·m 72. NOT 52/10.4% (that was the Total column).
- Structural zeros in shift training: 99 cells, 7 pop-sex, ages 95–99, all
  D=0, Jan-1 population 0 at both Lexis corners.
- RH bug story: cohort constraint documented-but-not-implemented → κ bent
  3.7 → coverage 0.68; fixed → nominal.
