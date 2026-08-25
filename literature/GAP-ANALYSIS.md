# Gap analysis — what the mortality-forecasting literature actually measures

Method: `pdftotext` over the library, then term-frequency counts plus manual reading of every hit.
**37 of 39** HMD-track PDFs extracted cleanly (`2102.09612` MFPCA and `2311.18668` LMM resisted extraction — likely image-only). 435k words searched.

This is evidence, not impression. Every claim below is a count or a quotation.

## 1. The headline counts

Across ~20 mortality-model papers in the library:

| Evaluation practice | Papers using it | Verdict |
|---|---|---|
| MSE / RMSE point accuracy | essentially all | the field's default |
| Interval coverage | **2** (Levantesi 2021; GBLL 2025) | near-absent |
| CRPS | **3** (Huynh–Ludkovski; Barigou BMA; vanishing-jump) | only in the Bayesian/GP strand |
| Log score / elpd | **2** (Barigou BMA; vanishing-jump) | rare |
| PIT / rank histogram | **0** | **absent** |
| Conformal prediction | **0** | **absent** |
| Joint / path coverage over horizons | **0** | **absent** |
| Diebold–Mariano or Model Confidence Set | **1** passing mention | **absent** |

**Neural mortality papers reporting any proper score: zero.**
**Neural mortality papers reporting interval coverage: one, and it is degenerate (see §3).**

## 2. The flagship neural papers are MSE-only

**Richman & Wüthrich (2021)**, the multi-population neural Lee–Carter that everything cites — term counts across the full text: `MSE` ×28, `mean squared error` ×3, `out-of-sample` ×11. `coverage` **0**. `CRPS` **0**. `log score` **0**. The model is fit to every country in HMD and evaluated purely on point error.

**Scognamiglio (2021)**, **Nigri et al. (2019)** (both the RNN paper and the Deep-Learning-Integrated-LC paper) — identical profile. Zero probabilistic metrics.

**Miyata & Matsuyama (2022)** is the sharpest case. The abstract sells "the ability to forecast with confidence intervals." Sections 6.4 and Figures 7–8 plot 97.5% confidence intervals for the latent factor and for mortality at ages 30–80. Then Table 3, the only comparison against Lee–Carter, is **MSE**. The paper draws the intervals and never once checks whether observed mortality falls inside them.

> They also note, of Schnürch & Korn (2021): *"The confidence intervals are not based on the endogenous randomness as in the BSS models, but on the exogenous randomness driven by the random seed for the NN."* — i.e. some published mortality "uncertainty" is just seed variance, which measures nothing about predictive coverage.

## 3. Where coverage IS reported, it is reported wrongly

**Levantesi et al. (2021)** — the single neural-LC paper that computes coverage. They use PICP (interval coverage probability) and MPIW (mean interval width) as **two separate indicators**, never combined into a width-penalised criterion. The consequence, in their own words:

> *"the LSTM offers always a greater probability coverage, in most cases due to the PI width"*
> *"The most virtuous example concerns the Australian males … the simultaneous presence of a total coverage of the future kt realizations and a proper PI width"*
> *"the LC-LSTM provides PICP(m) = 1"*

**PICP = 1.0 is treated as the best possible outcome.** It is not calibration; an interval wide enough to contain everything scores 1.0. Without a proper score or interval/Winkler score, "wider" and "better" are indistinguishable. Scope is also tiny — Australia, Japan, Spain; test window 2001–2018, i.e. **no structural break**.

And buried in the same paper, unchased:

> *"the ARIMA coverage probability … is around 50%, indicating that the predictive model fails, on average, to anticipate half of the future realizations"*
> *"the ARIMA coverage probability for Spanish males remains stable around 33%"*

**Nominal 95% Lee–Carter intervals attaining 33–50% empirical coverage, reported in passing, and nobody followed up.**

**Gradient-boosted multi-population (2025)**, the most recent competitor, repeats the pattern one level up: it reports marginal calibration (LL 90%, HBY 94%, GBLL 96% against a nominal 95%) and concludes GBLL is "the most properly calibrated" — again with no proper score and no sharpness penalty alongside.

## 4. The COVID stress test has never been run on real data

**Barigou, Goffard, Loisel & Salhi (2021)** is the closest paper methodologically: Bayesian Negative-Binomial, **leave-future-out validation**, log score (elpd) and CRPS. Genuinely good evaluation design. But its Section 6 COVID analysis is a **simulation** — they perturb French male deaths with "two years of excess mortality followed by one year of lower mortality" because in 2021 real post-shock data did not exist.

**It exists now.** From HMD's public What's New page, populations updated **through 2024**: USA, Japan, Denmark (2025), Belgium, Switzerland, Netherlands, Norway, Sweden, Finland, Portugal, Estonia, Latvia, Lithuania, Slovakia, Croatia, Iceland, Luxembourg, Korea, Taiwan, Hong Kong, Chile, Italy (2023), France (2023), Canada (2023), Spain (2023).

That is **20+ populations × 5 post-break years (2020–2024)** of real out-of-time data against a pre-registered train-through-2019 cutoff. The natural experiment the field has been simulating is now observable.

## 5. What this implies

The gap is not "add uncertainty to neural mortality models" — Levantesi and Miyata already produce intervals. The gap is threefold and each part is verifiable above:

1. **Neural mortality forecasting has never been evaluated with a proper scoring rule.** Not once, in any paper here.
2. **When coverage is reported it is not width-penalised**, so the literature's own conclusions about which model is "best calibrated" do not follow from its evidence.
3. **The one real structural break in modern mortality data has never been used as an out-of-time test** of these models — only simulated.

Plus four techniques entirely absent from mortality forecasting: PIT/rank histograms, conformal prediction, joint path coverage, and formal forecast-comparison testing (DM/MCS).

## 6. Papers to obtain before writing related work

- **Schnürch & Korn (2021)**, 2D-CNN mortality forecasting with confidence intervals — cited by Miyata as producing seed-based intervals. Not yet in the library; directly relevant to the "what counts as uncertainty" argument.
- **Perla et al. (2021)** SAJ — see `MISSING.md`.

---

# ADDENDUM (same day) — a correction to §1, and new prior art

Chasing the canonical actuarial baselines surfaced a body of work that the arXiv sweep had entirely missed, and it **partially invalidates the counts in §1**.

## Correction

§1 reported "PIT / rank histogram: **0** papers" and "joint / path coverage: **0** papers". Those counts were over *the library as it then stood*, which was arXiv-biased. They are **not** safe claims about the field. Specifically:

> **Dowd, Cairns, Blake, Coughlan, Epstein & Khalaf-Allah (2010),** *Backtesting Stochastic Mortality Models: An Ex-Post Evaluation of Multi-Period-Ahead Density Forecasts*, North American Actuarial Journal 14(3):281–298. `10.1080/10920277.2010.10597592`

The title alone establishes that ex-post evaluation of **multi-period-ahead density forecasts** for stochastic mortality models was done in 2010. This is direct prior art for the audit half of the paper and **must be read before any novelty claim is finalised.** It very likely uses PIT-type transforms.

Companion papers from the same group, now obtained:

- **Cairns, Blake, Dowd, Coughlan, Epstein & Khalaf-Allah**, *Mortality Density Forecasts: An Analysis of Six Stochastic Mortality Models* — in `pdf/`. Density forecasts across six models. Read this first; it is the closest obtained prior art.
- **Dowd et al. (2010)**, *Evaluating the Goodness of Fit of Stochastic Mortality Models*, IME 47:255–265 — not obtained.

## What this does and does not do to the idea

It **does not** kill the paper, but it moves the novelty. What survives, and what must be re-checked against Dowd/Cairns before committing:

| Claim | Status |
|---|---|
| Density-forecast backtesting for mortality is new | **DEAD.** Dowd et al. 2010 did it. |
| Proper scores (CRPS / log score) applied to *neural* mortality models | **Still open** — the 2010 work predates neural mortality entirely |
| Evaluation across a *real* structural break (2020–24), not a simulated one | **Still open** |
| Conformal prediction in mortality forecasting | **Still open** — nothing found in any source |
| Sharpness-subject-to-calibration critique of the PICP=1 practice | **Still open**, and now better grounded (Gneiting–Balabdaoui–Raftery 2007) |
| Crossed model-family × UQ-mechanism design | **Still open** |

**Method lesson, recorded deliberately:** the arXiv keyword sweep produced a systematically skewed view because ASTIN / AAS / NAAJ / IME largely predate and bypass preprint culture. Author homepages (Cairns' page at `macs.hw.ac.uk/~andrewc/papers/` yielded eight papers in one fetch) and Crossref are better instruments for actuarial literature than arXiv search.
