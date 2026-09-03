# Prior-art map against the five pre-registered hypotheses

**Dated 2026-08-26, committed and hash-stamped BEFORE any model has been fit to
real HMD data and before any real-data result exists.** That timing is the
point of this document: several verdicts below contradict the registered
direction of a hypothesis, and the amendments they force (registered in
`PREREGISTRATION-ADDENDUM-3.md`) have evidential value only if it is
demonstrable that they were made from the *literature*, not from our results.

Corpus: the 52 extracted texts in `literature/txt/`. Line references are into
those files (PDF extraction linebreaks shift by ±2 lines; quotes are verbatim
and were re-verified against the files on the date above). Where a source on
disk is a working paper, the citation names the published version.

---

## Verdicts

| Hypothesis | Verdict | One-line reason |
|---|---|---|
| H1 (RMSE vs proper-score ranking) | **PRIOR-ART-EXISTS** | Barigou et al. (2021) and Goes et al. (2024) already publish the conclusion; Schnürch–Korn's own Table 2 shows the inversion |
| H2 (universal under-coverage, worse for neural) | **PARTIAL — universality clause already falsified** in the stable regime | Schnürch–Korn: FFNN 97.0 / CNN 98.0 PICP vs 95 nominal; Levantesi et al.: LC-LSTM PICP = 1.0 |
| H3 (joint ≪ marginal coverage) | **NOVEL in mortality**; method not novel; registered comparator near-tautological | Dowd et al. name the dependence in a footnote and measure marginals anyway; 0 corpus hits for joint/path coverage in mortality |
| H4 (coverage failure concentrates at old ages) | **PARTIAL — registered direction contradicted** by prior art | Dowd et al. (published NAAJ): age 84 covers better than age 65. The sweeping "improve with higher ages" is **working-paper only** — see the provenance note below |
| H5 (miscalibration propagates to annuities/e_x) | **PARTIAL** — construction is prior art; ex-post coverage audit is NOT FOUND | Dowd et al. list it as their own un-executed extension |

---

## H1 — ranking by RMSE ≠ ranking by proper score

**PRIOR-ART-EXISTS.** Three independent instances:

1. Barigou, Goffard, Loisel & Salhi (2021), log score + CRPS + MAE side by side
   for LC/CBD/APC/RH/M6 across four countries:
   > "Concerning single models, there is no evidence of a best single model
   > across countries and the 'optimal' model depends on the country and the
   > scoring rule studied."
   — `2021_BayesianModelAveraging-Mortality-LeaveFutureOut__arXiv-2103.15434.txt:1079`
2. Goes, Barigou & Leucht (2024), Table 3 (England & Wales war-shock data,
   TRP 1901–1980, H=30): MSE/MAE/LogS rank AR first, **CRPS ranks MA first** —
   a published top-1 inversion, on a structural break.
   — `2023_Bayesian-Mortality-Pandemics-VanishingJump__arXiv-2311.04920.txt:688–757`
3. Schnürch & Korn (2022), Table 2: MSE ranks FFNN < CNN < ACF < LC10; Poisson
   deviance ranks LC10 < ACF < LC20 < RNN < CNN < FFNN — near-complete
   reversal. Their stated mechanism is exposure weighting (USA/Japan dominate
   the deviance), *not* distributional sharpness.
   — `SchnurchKorn2022_…ASTIN.txt:724–753`

**Consequence (registered in Addendum 3 §8):** H1 is demoted from finding to
harness-validity check. The one residual contribution: our CRPS/log-score
divergence must be shown to survive per-population reporting, i.e. not to be
Schnürch–Korn's population-size weighting artifact.

## H2 — nominal 95% intervals under-cover; degradation larger for neural

**PARTIAL.** The stable-regime half is settled prior art, and it contradicts
the registered universality clause:

- Schnürch & Korn (2022), Table 3 — 54 populations, ages 60–89, trained ≤2006,
  test 2007–2016, nominal 95%: **LC10 74.0 / LC20 74.3 / ACF 77.2 / RNN 86.0 /
  FFNN 97.0 / CNN 98.0** (PICP %); MPIW 0.012/0.011/0.010/0.015/0.017/0.019.
  > "The prediction intervals for the LC, ACF and RNN models ignore some
  > uncertainty as their realized coverage probabilities are significantly
  > smaller than 95%. Both the FFNN (97.0%) and the CNN (98.0%) fulfil the
  > requirement that PICP ≥ 0.95."
  — `SchnurchKorn2022_…ASTIN.txt:853–872`. Pass rule (one-sided):
  > "Good prediction intervals should minimize MPIW under the constraint that
  > PICP is at or above the specified threshold a. Here, we set a = 0.95."
  — `:850–851`
- Levantesi et al. (2021): classical interval coverage ~33% while the
  bootstrap-ensemble LSTM covers everything —
  > "period 1960-2000 for the Japanese females aged 65, the PIs from LC model
  > shows a coverage probability around 33%, while the LC-LSTM provides
  > PICP(m) = 1"
  — `Levantesi-etal2021_DeepeningLeeCarter-UncertaintyEstimation__arXiv-2103.10535.txt:686`

Two independent studies agree: in stable regimes, classical arms under-cover
and bootstrap-ensemble neural arms **over**-cover. "Under-cover for every
family" is therefore already falsified before our first fit, and our stable
regime is a **replication arm** and must be labelled as such.

**The shift half of H2 is the genuinely open part.** No paper in the corpus
evaluates mortality interval coverage out-of-sample across the real COVID
break, and the field has said why:
> "The pandemic predominantly impacts the most recent years of data, making it
> impractical to exclude these years and estimate parameters using only the
> earlier data. … Given these challenges, our analysis of the COVID data
> focuses solely on evaluating the in-sample fit of the models."
— Goes et al., `2023_Bayesian-Mortality-Pandemics-VanishingJump__arXiv-2311.04920.txt:379–380`
> "the validation technique could be adapted to the case where the mortality
> patterns exhibit a change of regime … However, this interesting problem is
> beyond the scope of the [paper]."
— Barigou et al., `2021_…arXiv-2103.15434.txt:1294`

With 2020–2024 now final for 20 HMD populations, that constraint is gone.
**Consequence (Addendum 3 §8):** H2 restated two-sided — |coverage − nominal|
grows under shift, direction free to differ by family.

## H3 — joint path coverage far below marginal

**NOVEL in mortality, as a measurement.** The field's own backtesting
framework names the dependence and then measures marginals anyway:
> "Note though that the positions of the individual observations within the
> prediction intervals are not independent. If, for example, the 1990
> observation is 'low', then the 1995 observation is also likely to be 'low'."
— Dowd et al. (2010), **footnote 10 of the published NAAJ article**
(`DowdCairnsBlakeEtAl2010_Backtesting-…NAAJ.txt:477`); the same sentence is
footnote 9 of working paper PI-0803
(`…NAAJ-PI0803.txt:699–701`, repeated at footnote 17, `:1681–1684`).
Verified in the published text 2026-09-03 — cite the footnote by its
published number. Corpus-wide grep for *joint coverage | simultaneous
coverage | path coverage | joint prediction region/interval | Bonferroni*
returns **zero hits in any mortality text**.

**NOT novel as a method.** Joint multi-horizon conformal coverage is solved
prior art in ML: CF-RNN (Stankevičiūtė et al. 2021, Bonferroni) and
copula-conformal (`2022_CopulaConformalPrediction-MultiStepTimeSeries__arXiv-2212.03281.txt:23,40`).
Wording discipline: *"first measurement of the marginal-to-joint coverage gap
in mortality forecasting, and the first across a structural break"* — never
language that reads as inventing joint coverage.

**Registered-form defect.** H3 as registered compares joint coverage to the
marginal rate at the same nominal level. Joint coverage over h=1…5 is bounded
above by the minimum marginal, and 0.95⁵ = 0.774 under independence, so the
registered comparison is close to arithmetically guaranteed to come out
"confirmed". **Consequence (Addendum 3 §8):** comparator re-specified as each
model's own simulated-path-implied joint coverage — computable from the same
sample paths, falsifiable, and the actuarially meaningful quantity.

## H4 — coverage failure concentrates at old ages

**PARTIAL, direction contradicted.** Coverage/accuracy-by-age curves are
already published for multiple families, and the prior finding is the
opposite of the registered direction:
> "Forecast performance tends to improve with higher ages."
— Dowd et al., **working paper PI-0803 only**,
`DowdCairnsBlakeEtAl2010_Backtesting-…NAAJ-PI0803.txt:728`.

**PROVENANCE CORRECTION (2026-09-03, on obtaining the published NAAJ text.)**
That sentence is a bullet in the working paper's summary and **does not
appear in the published article**, which plots two ages only and states the
narrower finding:

> "the age 65 forecasts have a notable upward bias with the realized values
> often close to or below the lower bounds, whereas the age 84 forecast show
> only a very slight upward bias."

— Dowd et al. (2010), NAAJ,
`DowdCairnsBlakeEtAl2010_Backtesting-…NAAJ.txt:487`. The direction of the
prior finding is unchanged (the higher age behaves better, so the registered
H4 direction is still contradicted), but the evidence is two ages and a bias
statement, not a monotone trend across the age range. `02-related-work.tex`
quoted the working-paper sentence against the published citation until this
was caught; it now quotes the published wording.
> "The ACF and LC models are very unreliable at the boundaries of the
> considered age range, while the NN approaches are more stable across ages."
— Schnürch & Korn (2022), Figure 7 discussion, `SchnurchKorn2022_…ASTIN.txt:873–876`

Also the ex-ante analogue: Cairns et al. find M1/M2 "imply that forecasts of
mortality at age 85 are much less uncertain than at age 65, contrary to
historical evidence"
(`DowdCairnsBlakeEtAl2010_MortalityDensityForecasts-SixStochasticModels.txt:509`).

What is ours: all prior work is confined to ages ~60–89. The 0–24 and 25–64
bands, infant mortality, and the 90+ tail are unmeasured, and no prior work
crosses age with a structural break. **Consequence (Addendum 3 §8):**
pre-commit to reporting a reversal of H4's registered direction as an
informative result, with the Dowd quote as the ex-ante alternative.

## H5 — miscalibration propagates to annuity factors and e_x intervals

**PARTIAL.** Constructing predictive distributions for annuity values and
life expectancy is established (nine-model annuity PV comparison in
`DowdCairnsBlakeEtAl2010_MortalityDensityForecasts-SixStochasticModels.txt:805`;
pointwise annuity-price intervals in `2020_MultipleFunctionalTimeSeries-…arXiv-2001.03658.txt:1489–1491`).
**Auditing the ex-post coverage of those intervals against realized outcomes
is NOT FOUND anywhere in the corpus** — and Dowd et al. name it as their own
un-executed extension:
> "Possible metrics include the mortality rate, life expectancy, future
> survival rates, and the prices of annuities and other life-contingent
> financial instruments. … In this paper, we focus on the mortality rate
> itself, but, in principle, backtests could be conducted on any of these
> other metrics as well."
— `DowdCairnsBlakeEtAl2010_Backtesting-…NAAJ-PI0803.txt:222–227`

**Risk, pre-registered as the alternative hypothesis:** prior evidence points
toward attenuation, not amplification —
> "We can see that there are only moderate differences between the models. …
> although the models can give quite different mortality forecasts, these
> differences can be attenuated when used in applications."
— `DowdCairnsBlakeEtAl2010_MortalityDensityForecasts-SixStochasticModels.txt:805`
(robust at 2% and 10% interest, `:806`). Note the quantities differ: Cairns et
al. describe dispersion of point applications across models; H5 tests
propagation of *calibration error*. Addendum 3 §8 pre-specifies that
distinction and names attenuation as the alternative being tested.

---

## The actual white space (the paper's spine)

1. **The crossed model-family × UQ-mechanism design.** The closest prior art
   states in its own words that it held the mechanism fixed:
   > "For calculating prediction intervals in our NN models, we rule out
   > several existing methods from the literature because they would require
   > substantial structural changes, such as the insertion of dropout layers
   > for Monte Carlo dropout … a change of the loss function for lower upper
   > bound estimation … or a substantial increase of the output dimension for
   > mean–variance estimation … We prefer not to change these hyperparameters
   > because doing so might decrease the forecasting performance, the
   > optimization of which is still our main goal."
   — Schnürch & Korn (2022), `SchnurchKorn2022_…ASTIN.txt:376–383`
   Their 74%-vs-98% coverage gap is therefore irrecoverably confounded between
   architecture and mechanism. Separating those factors is this paper's
   contribution, and this quote is its justification.
2. **Conformal arms in mortality.** Corpus-wide: "conformal" = **0 hits** in
   any mortality text; "MC dropout"/"deep ensemble" = 0 hits in any mortality
   text. The split/EnbPI/copula-conformal arms are unoccupied territory.
3. **H3's marginal-to-joint gap**, measured (see above).
4. **H5's ex-post derived-quantity coverage audit**, measured (see above).
5. **The out-of-sample pandemic test itself** — declined by the field for lack
   of post-break data (Goes et al. quote above), now feasible.

## Factual corrections to our own earlier notes

- Dowd et al. (2010a) backtest **six** models — M1, M2B, M3B, M5, M6, M7, each
  in parameter-certain and parameter-uncertain variants (`:149–154, :313–322`);
  M4 and M8 were dropped in the predecessor papers. Not "M1–M8".
- Schnürch–Korn's LC arms see only 10 or 20 training years while their NNs see
  all years and all 54 populations (`:591–594`). Their classical coverage
  numbers are partly a training-window artifact. Our expanding-origin rule
  gives every family the same window; the paper must say so explicitly.

## Known-unread (fetch before the affected section is written)

1. **Schnürch–Korn Online Supplementary Material, Section C.3** (implied
   annuity present values; referenced at `SchnurchKorn2022_…ASTIN.txt:1000–1004`;
   doi 10.1017/asb.2021.34). The only live threat to H5's residual novelty: if
   it audits annuity-interval *coverage* rather than presenting point PVs,
   H5's claim narrows further.
2. **Stankevičiūtė, Alaa & van der Schaar (2021), CF-RNN** — the correct
   baseline against which the copula-conformal arm must be positioned; cited
   in the corpus only second-hand.

---

## Addendum (2026-08-30) — the owed conformal-mortality search: white-space item 2 REFUTED

The idea-tournament critic demanded a web-scale search behind the claim that
conformal prediction had never been applied to mortality forecasting; the
verdict above ("conformal = 0 hits in any mortality text") was a corpus grep
over 52 files, not a literature search. The search has now been run. **The
field-wide claim is false and must never be used.** The corpus grep result
itself remains true of the corpus.

### Queries run (all on 2026-08-30)

| # | Channel | Query | Outcome |
|---|---|---|---|
| 1 | Exa web search | conformal prediction applied to mortality forecasting / Lee–Carter / life tables / actuarial prediction intervals | **3 direct hits** (Shang line of work, below) |
| 2 | Claude WebSearch | `"conformal prediction" "mortality forecasting" Lee-Carter` | no direct application beyond generic LC material |
| 3 | arXiv API (curl, https) | `all:"conformal prediction" AND (all:mortality OR all:"Lee-Carter" OR all:actuarial OR all:"life table")`, 30 newest | 2 direct hits (2605.29296, 2603.10674); remainder clinical-ML mortality classifiers, out of scope |
| 4 | Crossref API | `query=conformal prediction mortality forecasting` (20 rows) | no further application; surfaced `conformalForecast` CRAN pkg (Wang & Hyndman, general time series) |
| 5 | Crossref API | `query=conformal prediction Lee-Carter mortality` (15 rows) | nothing further |
| 6 | Crossref API | bibliographic query for Shang–Haberman | confirmed the SAJ record (below) |
| 7 | Exa web search | EnbPI / adaptive conformal in demography, life expectancy, death counts | same Shang papers; the "sequential conformal" they use is Xu–Xie SPCI, the EnbPI successor — so even the narrower "EnbPI-family conformal has not touched mortality" is refuted |

### What was found — the Shang (& Haberman) line, 2025–2026

1. **Shang & Haberman (2025), "Constructing prediction intervals for the age
   distribution of deaths", *Scandinavian Actuarial Journal***, published
   online 2025-08-15, doi 10.1080/03461238.2025.2544265 (arXiv:2506.17953).
   Split conformal intervals (absolute-residual quantile score) as one of two
   calibration approaches for life-table death counts, Japanese data, suite of
   functional time-series models. Appendices repeat the exercise on Australian
   and Canadian log mortality rates. **Predates our preregistration by a
   year, and is in one of our three target venues.**
2. **Shang (2026), "Conformal prediction for functional time series:
   application to age-specific mortality rates", *Journal of Population
   Research***, online 2026-06-04, doi 10.1007/s12546-026-09422-4
   (arXiv:2605.29296, 2026-05-28). Explicitly "our contribution is to
   introduce conformal prediction for modeling and forecasting age-specific
   mortality rates". Split + sequential (Xu–Xie SPCI) conformal on Australian
   age- and sex-specific log mortality 1921–2021, Hyndman–Ullah functional
   forecaster, ECP / CPD / mean interval score, expanding and rolling windows.
3. **Shang (2026), "Conformal prediction for high-dimensional functional time
   series: applications to subnational mortality"**, arXiv:2603.10674
   (2026-03-11). Same two conformal variants on Japanese and Canadian
   subnational log mortality 1975–2023; code at
   `github.com/hanshang/conformal_prediction_OWFANOVA_FFM`.

Adjacent, found and judged non-refuting: Hong (2023, conformal credibility)
and Chen et al. (2024, conformal cat-bond pricing) — actuarial, not mortality
forecasting; Duerst & Schöley (2024, *Pop. Health Metrics*) — empirical, not
conformal, intervals for short-term mortality; Wang–Hyndman `conformalForecast`
(CRAN 2025) — general multistep conformal, no mortality application. One
unresolved lead: "Shang & Hernandez (2025)" cited inside item 3 for
quantile-based intervals; not located, likely within the same line.

### What survives

None of the three papers: (a) audits conformal coverage across a designated
structural break (item 2's data end 2021, item 3's 2023; neither isolates
COVID as a test regime, and both evaluate in the Gibbs–Candès "ordinary
years" sense); (b) places conformal beside native, bootstrap, ensemble, or
dropout mechanisms — each uses a single functional-forecasting family with
conformal as the only calibration device, i.e. exactly the confounding the
crossed design exists to remove; (c) measures joint path coverage; (d)
touches derived quantities (e_x, annuities). The grid is untouched: the
conformal arms were registered as *audited arms*, never justified by being
first.

### Verdict and safe wording

- **REFUTED, never use:** "conformal prediction has not previously been
  applied to mortality forecasting", or any "first application of conformal
  prediction to mortality" claim, marginal or otherwise.
- **Safe:** "Conformal prediction entered mortality forecasting only in
  2025–2026, through Shang and Haberman's introduction of split and
  sequential conformal intervals for functional mortality forecasting; we
  are aware of no prior work that audits conformal coverage across a
  structural break, compares conformal against other uncertainty mechanisms
  within a crossed design, or measures joint path coverage in mortality."
- `paper/sections/02-related-work.tex` §"Conformal prediction" rewritten
  accordingly on 2026-08-30 (cites all three; bib keys `shang2025intervals`,
  `shang2026conformal`, `shang2026subnational`), and "the conformal arms" in
  §"What is and is not prior art" narrowed to the arms inside the crossed
  design and across the break.

### Incidental finding, for the H3 owner

**Li & Chan (2011), "Time-simultaneous prediction bands: a new look at the
uncertainty involved in forecasting mortality", *IME* 49(1):81–88, doi
10.1016/j.insmatheco.2011.02.006** (companion SOA Living-to-100 essay: Li &
Chan, US/Canada). Constructs *ex-ante, model-implied* simultaneous bands for
LC and CBD trajectories (Kolsrud adjusted intervals + Chebyshev bands) and
demonstrates that pointwise bands understate trajectory uncertainty. This is
not in the 52-text corpus and it weakens the H3 sentence "the field names the
dependence and measures marginals anyway": the field has *constructed* joint
bands. What it does not do is measure realized joint coverage against
outcomes (no backtest, no break). H3's safe wording is unchanged in substance
— "first *measurement* of the marginal-to-joint coverage gap … first across a
structural break" — but Li & Chan must be cited wherever joint bands are
discussed, and the Addendum-3 comparator (model-implied joint coverage) should
acknowledge that its construction is essentially theirs.
