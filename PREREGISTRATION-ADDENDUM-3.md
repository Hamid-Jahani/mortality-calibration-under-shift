# Pre-registration addendum 3 — data-defect handling, scoring-target repairs, and hypothesis-status clarifications

**Registered: 2026-08-26, before any model has been fit to real HMD data and
before any real-data result exists.** Hash-stamped in its commit like the
original, addendum 1 and addendum 2. Cross-references `docs/PRIOR-ART.md`
(sha256 `3f36fb62615deedb1dea9505e313581ace8ed3be3eb4962c63cdfd58164bdd12`,
committed immediately before this addendum), whose verdicts drive §8.

Changes no population list, no split year, no horizon, no metric list, and no
test-window observation. §1–§7 and §9–§11 specify computation on cases the
original text already covers; §8 records hypothesis-status clarifications
forced by prior art, made while no result exists.

## §1 — Zero-exposure training cells: weight matrix W = 1{E > 0}

`PREREGISTRATION.md` registers ages 0–99 and itself records that "cells with
E=0 are all at ages ≥90 in this vintage". Audit of the 2026-06-15 vintage:
the SHIFT training panel (20 populations × 2 sexes, years ≤ 2019, ages 0–99)
contains **99 cells with E = 0.00, every one with D = 0.00**, ages 95–99, in 7
(population, sex) series: CHE·f 1, CHE·m 10, FIN·m 27, ISL·f 13, ISL·m 30,
LUX·m 6, SWE·m 12. Verified structural zeros: HMD's Jan-1 `Population` count
is 0.00 at **both** corners of every one of the 99 Lexis squares — nobody was
alive at that age; nothing was suppressed.

Under the registered Poisson likelihood with log-exposure offset, such a cell
has mean λ = E·exp(η) = 0 and observation D = 0: its log-likelihood
contribution, score and Fisher information are all identically zero. The MLE
with the cell included equals the MLE under indicator weights **exactly**; a
numerical evaluation of 0·log 0 is the only thing that fails. **Rule:** every
family carries cell weights `W = 1{E > 0}` (StMoMo's `wxt` convention). This
is the numerically correct evaluation of the registered objective — a
clarification, no deviation flag.

Per-family specification:

- **Poisson-LC (PLC), Renshaw–Haberman:** weighted Poisson likelihood; W=0
  cells contribute nothing to Newton/IWLS steps. (RH already carries a weight
  matrix; PLC gains one.)
- **LC (SVD):** weighted rank-1 fit by alternating least squares / EM-SVD on
  centred log rates, with the exact unweighted SVD as a fast path when
  `W.all()` — the two must agree to machine precision at unit weights.
- **CBD (M5):** per-year weighted least squares on logit q at ages ≥ 55; the
  unweighted column-mean shortcut for κ1 is replaced by the weighted solve.
- **Sparse VAR:** common-complete-subsample — each fit uses regression rows
  built from consecutive year triples in which every age of the banded window
  is observed. Σ stays PSD by construction, OLS stays unbiased; no PSD
  projection is applied (a projection would perturb predictive width, the
  audited quantity, by an unquantified amount). Measured row loss (years
  ≤2019): CHE·f 142→139, CHE·m 142→125, FIN·m 140→101, ISL·f 180→164,
  ISL·m 180→136, LUX·m 58→49, SWE·m 267→242. All remain far above the banded
  VAR's minimum; every defective year is pre-1970.
- **Reduction requirement (tested):** every weighted path reproduces the
  unweighted fit bit-identically on panels with no zero-exposure cells.

Zero-death cells with E > 0 are unaffected by this clause: they are ordinary
observations under the Poisson likelihood and remain in every fit.

## §2 — Training-year contiguity; Belgium

The runner refuses training panels whose year columns are not contiguous
(previously only ages and test years were checked). **Rule:** the training
window for every (population, sex) is the maximal contiguous block of
complete years ending at the origin. Binding case: BEL, whose 1914–1918
deaths and exposures are missing at every age (1,000 cells — the only
genuinely missing data in the SHIFT training panel); BEL therefore trains on
**1919–2019** in the shift regime. Without this rule the pivot silently
deletes the five missing year-columns, a random-walk drift treats 1913→1919
as one step, and the RH cohort index (positional) mislabels every cohort
after 1913 by five years. BEL remains excluded from the placebo regime as
already registered.

## §3 — Zero-exposure cells in a TEST window

The placebo test window (1914–1922) contains E=0 test cells at ages 98–99:
CHE·m 1, FIN·m 3, ISL·f 5, ISL·m 1. No observation exists for such a cell
(nobody was alive). **Rule:** scoring masks ages exactly as the runner's age
mask already does — an age is scored only if every horizon's observation is
finite — with `n_ages_scored` reported and `cov95_by_age` null on masked
ages. Derived quantities (e0, e65, ä65) are computed on the maximal
contiguous scored age range starting at age 0, on both the predictive and the
observed side symmetrically, with the range reported; the life-table
truncation-invariance test already registered for CBD covers this path.

## §4 — Minimum training length (admissibility, all regimes)

Cells with very short training windows produce plausible-looking junk (a
random-walk σ from 2 differences; measured: CHL male, origin 1994, n_train=3
writes LC/native coverage_95 = 0.784 as a valid row). **Rule:** a (origin,
population, sex) triple is admissible only with **n_train ≥ 15** complete
training years. Inadmissible triples are recorded as such, not scored, and
the per-origin effective cluster count is reported alongside every
STABLE-regime summary (the stable panel is unbalanced: 27→33 buildable
pop-sex cells from origin 1990 to 2014).

## §5 — Rate-scale truth uses rounded deaths; the life-table convention split

Addendum 2 §1 made the rate-scale predictive samples integer-lattice
(Poisson-inclusive). Scoring fractional Lexis-split observed deaths against
an integer lattice makes a perfectly calibrated forecast look miscalibrated
(measured PIT KS 0.019→0.133 at λ=2; cov95 −2.2pp at λ=5) — a false-positive
generator aimed at H2/H4. **Rule:** the rate-scale observed truth is
`log(max(round_deaths(D), 0.5)/E)`, applying the already-registered half-up
rounding to the observed side so both sides of every rate-scale score live on
the same lattice. Comparability note: HMD death counts are non-integer for 9
of 20 shift populations (JPN/USA ~100% of cells, KOR 86–96%, …) and exactly
integer for the other 11; the per-population non-integer share is reported,
and the registered count-scale CRPS sensitivity remains the guard.

The observed **life table** (H5's `*_obs` quantities) keeps raw fractional
deaths and the 1e-10 rate floor: at ages where nobody died the crude-rate
floor is the correct life-table statement (m=0), while the half-count would
shift ISL·f 2020 e0 by −0.370 y — beyond gate 3's 0.15 y tolerance. The two
conventions — 0.5 deaths on the rate scale, 0 deaths in the life table, same
cell — are deliberate and are documented here.

## §6 — Conformal cells: scored from their interval bounds; calibration on the corrected scale

Addendum 2 §3 said conformal coverage/Winkler come "from the interval bounds
themselves". **Rule (implementation of that sentence):** each conformal
wrapper exposes its interval `[center − r, center + r]` at its construction
level 1−α (the wrapper's own α, not a hard-coded 0.95), and coverage/Winkler
for conformal cells are computed from those bounds directly — no
uniform-in-interval sampling, and **no Poisson composition** (the conformal
radius is calibrated on observed residuals and already contains observation
noise; composing more would double-count it — measured +150.7% width at
λ=10). CRPS/log score/PIT for conformal cells remain flagged secondary as
registered. Conformal calibration residuals use the same half-count
convention as the scorer (`log(max(D,0.5)/E)`), replacing the 1e-10 floor
whose single zero-death calibration cell inflated a SWE·f band radius from
1.44 to 12.94 nats; `_conformal_quantile` ignores non-finite residuals.

## §7 — SVAR: explosive coefficient draws are rejected, not clipped

The per-path OLS coefficient draws carry no stationarity constraint; on real
panels this produces divergent VARs with predictive m_x up to 3×10⁴⁵ and a
hard crash in the Poisson composer. **Rule:** a coefficient draw whose
companion-matrix spectral radius is ≥ 1 is rejected and redrawn (cap 100×
per path; a path that cannot produce a stable draw makes the cell an error
row). Predictive rates and Poisson means are **never** clipped to rescue a
divergent draw — clipping would convert divergence into a bounded interval
and silently flatter SVAR's coverage.

## §8 — Hypothesis-status clarifications (literature-driven; see docs/PRIOR-ART.md)

Made while no real-data result exists, from the prior-art map committed
immediately before this addendum:

- **H1** is demoted from headline hypothesis to **harness-validity check**:
  Barigou et al. (2021), Goes et al. (2024) and Schnürch–Korn (2022, Table 2)
  already publish the point-vs-proper-score ranking divergence. Our residual
  claim is magnitude under pre-registration, per population (not
  exposure-weighted).
- **H2** is restated **two-sided**: |empirical coverage − nominal| increases
  from the stable to the shift regime, with direction free to differ by
  (family, mechanism). The registered one-sided "under-cover for every
  family" is already falsified in the stable regime by published work (FFNN
  97.0 / CNN 98.0 PICP; LC-LSTM PICP = 1.0). The stable regime is a
  replication arm and will be labelled as such.
- **H3**'s comparator is re-specified: realized joint path coverage is
  compared to **each model's own simulated-path-implied joint coverage**
  (computable from the same predictive sample paths), not to the marginal
  rate at the same nominal level — the registered comparison is bounded
  toward confirmation (0.95⁵ = 0.774 under independence) and would be
  near-unfalsifiable. The gap realized-minus-implied can be positive, zero,
  or negative for any family.
- **H4**: the registered direction (worse coverage at 65–99 than 25–64) is
  contradicted by Dowd et al.'s published finding that forecast performance
  improves with age. We pre-commit to reporting a reversal as an informative
  result, not to re-testing until confirmation.
- **H5** is restricted to **ex-post coverage of derived-quantity intervals**
  (e0, e65, ä65) — the construction of such intervals is prior art. The named
  alternative hypothesis is Cairns et al.'s attenuation finding; H5 tests
  propagation of calibration error, not dispersion of point applications.

## §9 — Conformal calibration horizon covers the placebo

The placebo regime scores h = 1…9 while the conformal wrappers calibrated 8
horizons, so h=9 reused the last calibrated radius — uncalibrated by
construction on the regime where H3 has its longest path. **Rule:** the
calibration horizon is ≥ the longest scored horizon in every regime
(`cal_years`/`h_cal`: 8 → 9). The shift regime (h ≤ 5) is unaffected.

## §10 — Zero-death-cell reporting

The registered count of zero-death cells is per (population, sex, regime),
computed on the full age range 0–99 from the observed test window **before**
any model age mask, with the criterion `D < 0.5` (the threshold at which the
half-count correction binds; HMD fractional deaths make `== 0` under-count by
43% in the placebo). The per-cell column `n_zero_death_cells` follows the
same criterion. Corrected figures for the record: ISL·f 117/500 (23.4%),
ISL·m 68/500, LUX·f 110/500 (22.0%), LUX·m 72/500 in the 2020–2024 window —
the previously cited 52/10.4% was computed on HMD's `Total` column, which the
study never models.

## §11 — Family × mechanism contrasts are computed on common cells

Conformal arms fail on short panels (KOR: 17 training years fails every
conformal cell) and native arms fail where fits diverge — different cells,
by mechanism. Any contrast between mechanisms (or families) is therefore
computed on the **intersection** of (origin, population, sex) cells in which
every compared arm produced a valid row, with the intersection size reported;
full uncensored tables go to the appendix. Error rows are never silently
dropped from a denominator: every table reports its cell count.

---

Nothing in this addendum touches a test-window observation, a split year, a
population list, or the metric list. §1–§7, §9–§11 specify computation; §8
records literature-forced hypothesis-status clarifications with the prior-art
map as dated evidence.
