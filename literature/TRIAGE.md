# Library triage — what actually earns its place

Assessed against the locked idea: **audit + conformal fix, HMD, calibration under the COVID break**.
Honest verdict: of 48 HMD-track PDFs, about **30 earn their place**. The rest are context, misfiled, or keyword-sweep noise.

## Tier 1 — CORE (17). Read fully. Load-bearing for the contribution.

| Paper | Its job in this paper |
|---|---|
| Lee & Carter (1992) | The baseline. Note it *already* forecasts with intervals — our whole question is whether the successors kept that. |
| Girosi & King, *Understanding Lee–Carter* | LC ≈ multivariate RWD; forecast age-profiles degrade. Prevents a naive baseline claim. |
| Richman & Wüthrich (2021) neural LC | The benchmark model **and** the evidence: `MSE`×28, coverage×0. |
| Scognamiglio (2021) | Neural LC / Poisson-LC calibration. Baseline + MSE-only evidence. |
| Miyata & Matsuyama (2022) VAE-LC | Baseline with native intervals **and** the evidence they are never validated. |
| Levantesi et al. (2021) | The `PICP = 1` methodological error + the 33–50% LC coverage smoking gun. |
| Nigri et al. (2019) *Risks* | LSTM-on-k_t baseline. |
| Barigou et al. (2021) BMA leave-future-out | Best existing eval design; the *simulated*-COVID precedent we replace with real data. |
| Huynh & Ludkovski (2021) MOGP | Probabilistic comparator; the one mortality paper using CRPS properly. |
| Gradient-boosted multi-pop (2025) | Most recent competitor; reports coverage without a sharpness penalty. |
| **Gneiting, Balabdaoui & Raftery (2007)** | *"Maximise sharpness subject to calibration."* The theoretical basis for why `PICP=1` is wrong. **The single most important methodology citation in the paper.** |
| **Gneiting & Raftery (2007)** | Proper scoring rules — CRPS, log score, interval score definitions. |
| Murphy decomposition (2020) | Splits a bad score into calibration vs resolution. Says *why*, not just *that*. |
| Xu & Xie, EnbPI | Conformal intervals for time series without exchangeability. Part 2 workhorse. |
| **Gibbs & Candès (2021) ACI** | Adaptive conformal inference *under distribution shift*. The method Part 2 is built on. |
| Copula-conformal multi-step (2022) | Joint path coverage over h=1…H. |
| Lakshminarayanan et al., Deep Ensembles | The neural UQ mechanism being audited. |

## Tier 2 — NEEDED (12). Skim; cited in related work or methods.

Barber et al. *Conformal beyond exchangeability* (justifies conformal on mortality time series) · Gibbs & Candès (2022) online arbitrary shifts · Tibshirani et al. covariate shift · Romano et al. CQR · Conformal TS with change points (2025) · Angelopoulos & Bates primer · **MCS + forecast combination for age-specific mortality (2018)** — the one prior use of Model Confidence Set in this exact setting · **StMoMo JSS paper** — the R oracle's reference implementation · Nigri et al. (2019) RNN-vs-LC · COVID-19 shock on multi-pop model (2021) · Bayesian pandemics vanishing-jump (2023) · Bayesian Poisson log-normal multi-pop (2020).

## Tier 3 — CONTEXT (8). One-line citation. Do not read.

Catastrophe risk multi-pop · COVID granular impact · GAM+ML COVID trend · Granular regime-switching (2025) · Actuarial learning pension-fund mortality (2025) · State-level longevity USMDB · Semi-parametric multi-pop · Matrix-distribution mortality · Deprez/Shevchenko/Wüthrich ML mortality · Proper scoring rules for survival analysis.

## Tier 4 — DEAD WEIGHT for this paper (5). Keyword-sweep noise.

- The **functional-time-series cluster** — MFPCA time-weightings, multiple FTS group structure, high-dim subnational FTS, LMM mortality. Legitimate mortality forecasting, but they are *point*-forecast functional methods that would not be baselines here and do not bear on calibration. Two of them would not even extract to text.
- `RichmanWuthrich2019_NeuralLeeCarter-SLIDES` — redundant with the full preprint. Keep only for architecture figures.

## Tier 5 — MISFILED (3). Belong to the deferred explainability paper, not this one.

LocalGLMnet · Brauer transformers pricing · Multi-task discrimination-free pricing. These are pricing/interpretability, not mortality. Move to `pdf-other-datasets/` or a `paper2/` folder.

---

## The real problem: what is MISSING matters more than what is surplus

The library is strong on *recent arXiv mortality papers* and weak on *canonical actuarial baselines*, because those are pre-arXiv and paywalled. Every model in the roster below needs its defining reference before it can be implemented and cited:

| Missing | Needed for | Status |
|---|---|---|
| **Brouhns, Denuit & Vermunt (2002)** | Poisson Lee–Carter — the primary baseline and the source of the semiparametric bootstrap | Not obtained |
| **Cairns, Blake & Dowd (2006)** | CBD / M5 — the old-age baseline | Not obtained |
| **Renshaw & Haberman (2006)** | APC / cohort baseline | Not obtained |
| **Cairns et al. (2009)**, quantitative comparison of stochastic mortality models | The standard model-comparison protocol in actuarial mortality | Not obtained — often free from the Pensions Institute |
| **Perla, Richman, Scognamiglio & Wüthrich (2021)** | Shallow-CNN LC baseline | SSRN, browser-only — see `MISSING.md` |
| **Schnürch & Korn (2021/22)** | 2D-CNN with intervals; the "uncertainty = random seed" argument | Cambridge paywall |
| Diebold & Mariano (1995); Hansen, Lunde & Nason (2011) | Forecast-comparison testing | Not obtained (classic, easy to cite from text) |
| Cairns et al. (2020) COVID accelerated-deaths | Pandemic-year treatment | Not obtained |

**Assessment:** the collection is over-weighted toward things that were easy to download and under-weighted toward the four or five actuarial papers that define the baselines we must implement. Fixing that is worth more than pruning Tier 4.
