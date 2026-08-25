# Pre-registration addendum 2 — scoring-target clarifications

**Registered:** 2026-08-26, before any model has been fit to real HMD data.
Clarifies how two pre-registered metrics are computed; changes no
hypothesis, population, split, horizon, or metric list. Hash-stamped in its
commit like the original and addendum 1.

## 1. Rate-scale scores evaluate the predictive law of the OBSERVED rate

`PREREGISTRATION.md` registers CRPS, Winkler scores, coverage and PIT "on
log m_x". The observed m_x = D/E carries Poisson sampling noise; a model's
latent-rate samples do not. Scoring latent samples against observed rates
would count observation noise as miscalibration, most severely at young ages
and in small populations where expected deaths per cell are small.

Rule: every rate-scale score is computed on **Poisson-inclusive** predictive
samples, D* ~ Poisson(E · m_x*) drawn on the model's own m_x* paths, converted
to log crude rates. This is the predictive distribution of the quantity that
is actually observed. It is the same construction already registered for the
death-count metrics (log score, CRPS on counts). Model-native, bootstrap,
ensemble and conformal mechanisms are all treated identically, so the crossed
design is unaffected.

## 2. Zero-death cells: half-count continuity correction, no masking

Cells with zero observed deaths have no finite log rate. Rule: log rates —
observed and sampled alike — use max(D, 0.5)/E, the standard demographic
continuity correction; the same convention is applied to the predictive
samples so both sides of every score share it. Cells are NOT dropped; the
number of zero-death cells per (population, sex, regime) is reported.
Count-scale metrics (Poisson log score, CRPS on counts) need no correction
and use the unrounded / half-up-rounded deaths exactly as registered.

## 3. Conformal cells: scored on their intervals, at their construction level

Conformal mechanisms yield intervals, not distributions. Rule: coverage and
Winkler scores for conformal cells are computed from the interval bounds
themselves at the construction level (95%); the 50% and 80% columns are
reported as not-applicable for these cells, and CRPS/log score/PIT remain
flagged secondary as already documented in `mortcal/uq/conformal.py`. This
removes a scoring bias against conformal arms that would otherwise arise
from taking empirical quantiles of uniform-in-interval samples.

## 4. PIT uniformity: statistic and a caveated p-value

The KS statistic is reported together with its nominal p-value; because PIT
values across ages and horizons within a population are dependent, the
p-value is descriptive. Formal inference on calibration uses the
population-clustered procedures already registered.

Nothing in this addendum alters what is being tested, only how the
registered quantities are computed from observed data.
