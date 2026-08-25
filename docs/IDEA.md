# The idea — tournament record and rationale

**Decision date:** 2026-08-25. **Method:** 15-agent workflow — 7 evidence agents
(3 auditing the actual HMD files on disk, 4 building literature dossiers) → 4
candidate designs from distinct lenses → 3 adversarial judges (actuarial
referee, ML-methods reviewer, pragmatist) → completeness critic.

## Winner

> **Do Mortality Prediction Intervals Survive a Pandemic? A Pre-Registered
> Calibration Audit of Ten Forecasting Families and Seven Uncertainty
> Mechanisms Across the COVID-19 Break**

**Claim.** Using the first HMD vintage with final single-age data through 2024
(20 populations), a pre-registered crossed audit of model family × uncertainty
mechanism shows whether nominal 95% mortality prediction intervals — classical,
neural, and conformal alike — hold coverage through the COVID break; failures
are resolved by age, by horizon, at the joint-path level, and in economic terms
(e65, ä65 intervals). The audit protocol (synthetic-truth-validated harness,
oracle-parity baselines, proper scores + PIT + joint path coverage + DM/MCS)
is itself a reusable contribution.

**Why it is a *study* and not a bake-off:** model family and uncertainty
mechanism are separate crossed factors, so "bad coverage" can be attributed to
the architecture or to the uncertainty machinery bolted onto it — a separation
no prior mortality paper makes.

## Final scores (3 judges, novelty + feasibility + venue-fit /30)

| Candidate | Mean | Verdict |
|---|---|---|
| **Do Mortality Prediction Intervals Survive a Pandemic?** (audit at scale) | **24.3** | **winner** — unanimous first on all three ballots |
| When the Fan Chart Fails (audit + conformal-fix headline) | 23.0 | two-papers-in-one; fix section first casualty of page limits; feasibility 5–6 |
| Conformal Mortality Forecasting (method-led) | 20.7 | venue risk: reads as ML import; guarantee language fragile under a 5-year break |
| Twin Crises, One Law? (1918→2020 transfer test) | 20.0 | highest novelty (9,9,8) but feasibility 4–5: WWI data quality, era-1 neural fits |

## What the losers contributed (absorbed at near-zero scope cost)

- From *Fan Chart Fails*: **NexCP** (geometric year-weights) and **CopulaCPTS**
  as UQ-mechanism levels inside the grid — audited arms, no "fix" claim.
- From *Conformal Mortality Forecasting*: conformal arms output **full
  calibrated CDFs** (conformal predictive systems), so every arm is
  CRPS-scorable through the single evaluation interface.
- From *Twin Crises*: a **descriptive 1918-vs-2020 side-by-side coverage
  table** using the placebo regime as-is (no transfer regression).

## Evidence base that drove the choice (verified, not assumed)

- Data on disk: 50 populations; deaths/exposures year-ranges identical; 20
  populations final through 2024; 14 placebo-eligible, BEL excluded (555
  missing cells 1914–18); 15,496 zero-exposure cells all at ages ≥90; our D/E
  reproduces HMD's published Mx exactly at spot checks.
- Cairns et al. 2011 (IME) density-forecast paper is **ex ante and
  qualitative** — fan charts judged by "biological reasonableness"; no PIT, no
  coverage computation, no proper scores, no held-out evaluation.
- Levantesi et al. 2021: PICP=1.0 read as success, "high MPIW desirable";
  their own LC/ARIMA coverage of 0.33–0.56 reported without alarm.
- Miyata & Matsuyama 2022: intervals plotted (Figs 7–8), models compared by
  MSE only.
- No paper evaluates probabilistic mortality forecasts on real 2020–2024
  held-out data; the closest is a WWII-break log-score evaluation
  (Goes/Barigou/Leucht 2023) and a synthetic COVID perturbation (Barigou 2021).

## Critic's gates and how each is being handled

| Gate | Status |
|---|---|
| Read Dowd et al. 2010b before novelty text | **User fetching** (SSRN 1396201, browser-only). Introduction/related-work text blocked on it; experiments are not. If 2010b tabulated hit rates, the delta narrows to: real break + crossed design + joint path coverage + age resolution + proper scores on realized 2020–24 — still sufficient. |
| Read Schnürch & Korn 2022 | **User fetching** (SSRN 3796051). Plausibly computes neural interval coverage — the delta budget above already assumes it does. |
| Crossref/Scholar conformal-mortality search | Open TODO before submission; grep-based evidence is indicative only. |
| Fractional-deaths Poisson log score undefined | **Closed** — convention fixed in PREREGISTRATION.md (round to nearest integer; CRPS-on-counts sensitivity). |
| Ragged 10×7 grid vs factorial claims | **Closed** — docs/GRID.md enumerates 50 admissible primary cells; prereg commits to within-sub-grid contrasts only. |
| Effective n ≈ 1 common shock | **Closed** — prereg frames shift-regime inference as an event study: wild cluster bootstrap over populations, pooled PIT, placebo regime as second event. |
| 2024 values final or revisable? | Open TODO — check HMD revision policy; robustness plan already includes drop-2024 re-run. |
| WWI country-notes for placebo populations | Open TODO before placebo-regime results are read. |

**Deviation from the critic's ordering, recorded honestly:** the critic wanted
Dowd 2010b read *before* freezing PREREGISTRATION.md. The prereg was committed
first (2026-08-25, sha256 bbbe3860…) because it freezes *analytic choices*
(splits, metrics, hypotheses, populations) — none of which depend on Dowd's
content — and the COVID outcomes are public knowledge anyway, so
pre-registration here protects against analytic flexibility, not hypothesis
surprise (the critic's own point). What Dowd 2010b can change is the *framing
of novelty in the paper text*, which is not frozen.

## Where this leaves the folder name

"Mortality – Explainable AI": explainability is deliberately **not** in this
paper. It is the planned paper #2 (explanation stability/faithfulness of
neural mortality models), reusing this repo's fitted models and harness.
