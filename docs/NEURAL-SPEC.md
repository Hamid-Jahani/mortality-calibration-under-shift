# Neural / GP family specification — the missing five rows of GRID.md

**Written 2026-08-26, before any neural code exists.** Implements the five
registered families (PREREGISTRATION.md:62–65) and the two mechanisms that
exist only on them (deep ensemble M=10, MC dropout). Nothing here is a new
architecture — each family reimplements its published source, simplified only
where the per-(population, sex) panel (≤100 ages × ≤270 years) makes the
published multi-population apparatus inapplicable. The framing rule stands:
the contribution is the audit, not the models.

## Ground rules (binding, from the registered documents)

1. **One interface** (methodology rule 4): `fit(D, E)` →
   `sample_mx(h, n, rng) -> [n, h, n_ages]`. The runner's scoring path is
   untouched.
2. **Registered likelihood:** every count-modelling family trains on the
   Poisson deviance with `log(E)` offset (PREREGISTRATION.md:40) — never
   Gaussian on raw rates. Rate-modelling stages (GP, SVD stage of LSTM-k_t)
   use log crude rates on the half-count scale `log(max(D, 0.5)/E)`
   (addendum 2 §2), with `W = 1{E > 0}` cells excluded (addendum 3 §1).
3. **Hyperparameters tune on an inner TIME split** (methodology rule 3): the
   last 5 training years are the inner-validation block; the small fixed grid
   below is searched once per cell; the winner refits on the full training
   window. No test year is ever seen.
4. **Seeds recorded per member** (rule 7): every ensemble member / dropout
   pass / torch init derives from the cell's `SeedSequence`
   (`runner._cell_seed`) by documented spawn index.
5. **Determinism:** torch seeded and `torch.use_deterministic_algorithms`
   where supported; fits are CPU-reproducible. GPU is an optimisation, not a
   dependency — at this model scale (thousands of parameters, ≤27k cells)
   CPU training is seconds per fit.
6. **Admissibility** is `docs/GRID.md`, transcribed into `runner.ADMISSIBLE`
   consciously, cell by cell. Native is inadmissible for neural-LC / CNN-LC /
   LSTM-k_t (point forecasters — no predictive law of their own); their
   `sample_mx` returns the DEGENERATE repeated point path, documented, whose
   only legitimate consumers are the conformal centre (median of identical
   paths = the point path) and point metrics. `run_cell` refuses inadmissible
   cells before any fit.

## The five families

### 1. `NeuralLC` — Richman–Wüthrich (2021) embeddings, per-panel form
Source: `literature/pdf/RichmanWuthrich…AAS` roster entry. The published model
embeds (age, year, country); fit per (pop, sex) here, the country embedding
drops. Features: age embedding (dim 5) ⊕ scaled calendar year. Trunk: 2 × 64
tanh, dropout p=0.05 after each hidden layer. Output: `log m̂` added to
`log E` offset → Poisson deviance loss. Adam; grid: lr ∈ {1e-2, 3e-3} ×
epochs ∈ {200, 500}. Forecast: evaluate at future years (year feature
extrapolates — this is the audited fragility, not a defect to engineer away).

### 2. `CNNLC` — shallow CNN, Perla et al. (2021) / Schnürch–Korn (2022) form
Input: trailing window of L=10 years × all ages of `log(max(D,0.5)/E)`
(missing cells filled with the age's trailing observed mean — fill value is
never a target). One Conv2d (8 filters, 3×3, ReLU) → flatten → linear to
n_ages outputs = next year's `log m̂`; Poisson deviance via the offset.
Multi-step: recursive, feeding predictions back. Grid: lr ∈ {1e-3, 3e-4} ×
epochs ∈ {300, 800} — **corrected 2026-08-27, before any real-data run**: the
originally specified {1e-2, 3e-3} diverges at every grid point on this loss
scale (measured in-sample RMSE 10–20 nats; 1e-3 converges to 0.12).

### 3. `LSTMKt` — LSTM on the Lee–Carter index
Stage 1: the EXISTING `LeeCarterSVD` fit (EM-SVD path included) supplies
(α, β, κ). Stage 2: LSTM (hidden 16, 1 layer) on the κ series, input window
10, trained to predict κ_{t+1}; innovation σ from one-step training
residuals (ddof=1). Forecast paths: recursive mean prediction plus fresh
N(0, σ²) innovation per horizon per path — the family carries innovation
noise exactly as RWD does; parameter/seed uncertainty is the MECHANISM's job.
Grid: lr ∈ {1e-2, 3e-3} × epochs ∈ {300, 800}.

### 4. `NBHead` — distributional negative-binomial head
Features as NeuralLC. Two outputs per cell: `log μ` (plus `log E` offset) and
`log r` (dispersion). Loss: NB2 negative log-likelihood. **Native sampling
composes with the runner's Poisson step by construction:** NB is a
Poisson–Gamma mixture, so `sample_mx` draws the GAMMA rate
`λ ~ Gamma(r, scale=μ/(rE))` — the runner's registered Poisson composition on
top then yields exactly the NB predictive count law. No double count, one
code path. Grid: lr ∈ {1e-2, 3e-3} × epochs ∈ {200, 500}.

### 5. `MultiOutputGP` — multitask GP over years, ages as tasks
Source: Huynh & Ludkovski (2021), reduced to one population. gpytorch exact
multitask GP: input = scaled year, tasks = ages, kernel = RBF(year) ⊗
ICM(rank 5), Gaussian likelihood on `log(max(D,0.5)/E)`. Kronecker structure
requires a complete year × age block: the GP trains on the TRAILING block of
complete years (every age observed), minimum 40 — for the affected
populations every incomplete year is pre-1970, so this is the same
common-complete principle as SVAR (addendum 3 §1), applied to the trailing
window. Native `sample_mx`: joint posterior samples over the h future years —
the GP posterior is the model-native law (GRID.md: bootstrap inadmissible
because the posterior already integrates parameter uncertainty). Grid:
lr ∈ {1e-1, 3e-2} × iters ∈ {200, 400} (marginal-likelihood training).

## The two mechanisms (`mortcal/uq/neural.py`)

### `DeepEnsemble` (M=10)
M members of the wrapped family, identical hyperparameters, member m seeded
from the cell SeedSequence spawn m (recorded). `sample_mx` distributes the n
requested paths over members (uniform with replacement) and draws each path
from the member's own `sample_mx` — for point families a mixture of (near-)
deltas, for NBHead a mixture of NB laws, exactly the Lakshminarayanan et al.
(2017) predictive. Known property, stated in advance: mixture-of-deltas
ensembles are narrow; if they under-cover, that is a FINDING about the
mechanism (H2), not an implementation defect to widen away.

### `MCDropout`
The wrapped family trains once with its dropout layers; `sample_mx` runs n
stochastic forward passes with dropout ACTIVE (Gal & Ghahramani 2016), plus
the family's own innovation noise where it has any (LSTMKt). Families without
dropout layers (GP, classical) are inadmissible — enforced in the registry.

## Validation gates (per family, before any real data)

Extends `tests/test_synthetic_calibration.py` discipline:
- **G-N1 (right scale; recalibrated 2026-08-27, measured):** on the
  synthetic Poisson-LC DGP each family's point forecast has finite
  RMSE(log m) < 1.0 over h=1..5. Persistence scores 0.25–0.51 on these
  worlds and the failure modes the gate catches measured 1.08–148; the
  cell-feature nets' extrapolation plateau (~0.5) is the documented audited
  fragility, so the original within-2×-of-PLC form was wrong for
  deliberately misspecified learners. Quality is measured by the study, not
  asserted by gates.
- **G-N2 (interface):** shapes, finiteness, determinism given seed, and
  degenerate-native documentation for the point families.
- **G-N3 (NB coherence):** NBHead's Gamma-rate sampling composed with the
  runner's Poisson step reproduces NB coverage on an NB-simulated world
  (nominal within tolerance at 50/80/95).
- **G-N4 (structural zeros):** every family fits through the addendum-3
  E = 0 test panels.
- **G-N5 (ensemble/dropout seeds):** member seeds distinct, recorded, and
  reproducible from the cell SeedSequence.

## Runner integration

- `MODELS` += {"GP": MultiOutputGP, "NLC": NeuralLC, "CNN": CNNLC,
  "LSTM": LSTMKt, "NB": NBHead}; `MECHANISMS` += ("ensemble", "dropout").
- `ADMISSIBLE` transcribed from GRID.md including the (s) secondary cells,
  with a `SECONDARY` frozenset so the runner can flag them.
- Conformal wrappers take the neural families as `base_factory` unchanged —
  their centre calls `sample_mx`, which the degenerate point path serves.
- torch is an OPTIONAL import: `mortcal` without the neural extra keeps the
  classical grid fully functional (guarded import, clear error naming
  `uv sync --group neural`).

## Out of scope (explicitly)

Architecture search beyond the fixed grids; multi-population/transfer
variants; any tuning against coverage (tuning is loss-based on the inner
split only — tuning UQ against coverage would make H2 circular).
