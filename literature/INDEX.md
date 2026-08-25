# Literature Library

39 PDFs in `pdf/`. Downloaded 2026-08-24. Naming: `Author/Year_ShortTitle__arXiv-ID.pdf`.

## A. Core benchmarks — classical mortality models

| File | Why it matters |
|---|---|
| `LeeCarter1992_...JASA.pdf` | The origin. SVD decomposition + RWD on k_t. Note: **it already forecasts with confidence intervals** — our paper asks whether the neural successors kept that property. |
| `GirosiKing_Understanding-the-LeeCarter-Method.pdf` | Critical: proves LC ≈ multivariate random walk with drift, and that forecast age profiles become non-smooth. Essential caveat for our baseline section. |

## B. Neural extensions of Lee–Carter (direct competitors / benchmarks)

| File | Why it matters |
|---|---|
| `RichmanWuthrich2021_NeuralLeeCarter-MultiPop_FULLPREPRINT.pdf` | **The** benchmark to reproduce. Representation learning over all HMD countries, 1950–1999 train / →2016 test. Point accuracy only — no calibration. That gap is our paper. |
| `RichmanWuthrich2019_NeuralLeeCarter-SLIDES.pdf` | Slide deck of the above; useful for architecture details. |
| `Scognamiglio2021_CalibratingLC-PoissonLC-via-NN__arXiv-2106.12312.pdf` | Joint NN calibration of LC / Poisson-LC across populations. Doc-cited. |
| `MiyataMatsuyama2022_LeeCarter-VAE-Bayesian_ASTIN_OA.pdf` | LC + VAE, single-stage variational Bayes, **produces confidence intervals without MCMC**. Closest existing work to our UQ angle. Open Access. |
| `NigriEtAl2019_DeepLearningIntegrated-LeeCarter_Risks.pdf` | LSTM on k_t inside LC structure. Open Access (MDPI Risks). |
| `Nigri-etal2019_RNN-vs-LeeCarter__arXiv-1909.05501.pdf` | "Can RNNs beat Lee–Carter?" — the point-accuracy framing we are deliberately moving past. |
| `Levantesi-etal2021_DeepeningLeeCarter-UncertaintyEstimation__arXiv-2103.10535.pdf` | LC deepening **with uncertainty estimation** — nearest neighbour to our contribution; read early to sharpen the delta. |
| `DeprezShevchenkoWuthrich2017_ML-Techniques-Mortality-Modeling__arXiv-1705.03396.pdf` | Tree-based ML for mortality; early actuarial-ML reference. |
| `2025_ActuarialLearning-PensionFund-MortalityForecasting__arXiv-2504.05881.pdf` | Recent; pension-fund framing. |
| `2025_GradientBoosted-MultiPop-Mortality-HighFreq__arXiv-2507.09983.pdf` | GBM multi-population baseline — non-neural ML comparator. |

## C. Probabilistic / Bayesian mortality comparators

| File | Why it matters |
|---|---|
| `HuynhLudkovski2021_MultiOutputGP-MultiPop-Longevity__arXiv-2003.02443.pdf` | Multi-output GP. Coherent joint forecasts with explicit uncertainty. Our principal probabilistic comparator. |
| `2020_BayesianPoissonLogNormal-RegularizedTime-MultiPop__arXiv-2010.04775.pdf` | Bayesian Poisson log-normal, regularized time structure. |
| `2021_BayesianModelAveraging-Mortality-LeaveFutureOut__arXiv-2103.15434.pdf` | **Leave-future-out validation** for mortality — directly informs our rolling-origin protocol. |
| `2020_MortalityModeling-Regression-MatrixDistributions__arXiv-2011.03219.pdf` | Phase-type / matrix distribution mortality models. |
| `2023_ForecastingMortalityRates-LinearMixedEffects__arXiv-2311.18668.pdf` | LMM comparator (short note). |

## D. Structural break / COVID — the stress-test literature

| File | Why it matters |
|---|---|
| `2021_COVID19-Shock-Stochastic-MultiPop-Mortality__arXiv-2111.10164.pdf` | Quantifies what the COVID shock does to a multi-pop stochastic model. Our shift-regime motivation. |
| `2023_Bayesian-Mortality-Pandemics-VanishingJump__arXiv-2311.04920.pdf` | Vanishing-jump treatment of pandemic years — a principled alternative to deleting 2020–21. |
| `2023_CatastropheRisk-Stochastic-MultiPop-Mortality__arXiv-2306.15271.pdf` | Catastrophe/jump risk in mortality. |
| `2022_COVID19-Impact-Granular-Mortality-Data__arXiv-2209.06473.pdf` | Granular COVID mortality impact estimation. |
| `2023_GAM-ML-COVID19-MortalityTrend__arXiv-2311.15401.pdf` | GAM + ML for COVID trend deviation. |
| `2025_GranularMortality-Temperature-EpidemicShocks-RegimeSwitching__arXiv-2503.04568.pdf` | Regime-switching under epidemic + temperature shocks. |

## E. Functional / high-dimensional time-series comparators

`2021_MultiPop-MFPCA-TimeWeightings__arXiv-2102.09612.pdf`,
`2020_MultipleFunctionalTimeSeries-GroupStructure-Mortality__arXiv-2001.03658.pdf`,
`2023_HighDim-FunctionalTimeSeries-SubnationalMortality__arXiv-2305.19749.pdf`,
`2020_SemiParametric-MultiPop-MortalityModel__arXiv-2009.04296.pdf`,
`2023_StateLevel-LongevityTrends-USMDB__arXiv-2312.01518.pdf`

## F. Evaluation methodology — scoring rules, calibration, conformal

| File | Why it matters |
|---|---|
| `2014_Theory-Applications-ProperScoringRules__arXiv-1401.0398.pdf` | Foundation for log score / CRPS choices. |
| `2020_MurphyDecomposition-CalibrationResolution__arXiv-2005.01835.pdf` | Decomposes a score into calibration + resolution — lets us say *why* a model scores badly, not just that it does. |
| `2023_ProperScoringRules-SurvivalAnalysis__arXiv-2305.00621.pdf` | Scoring rules where the target is a survival/lifetime distribution. |
| `LakshminarayananEtAl2017_DeepEnsembles-PredictiveUncertainty__arXiv-1612.01474.pdf` | Deep ensembles — our cheapest credible neural UQ mechanism. |
| `AngelopoulosBates2021_GentleIntro-ConformalPrediction__arXiv-2107.07511.pdf` | Conformal primer. |
| `XuXie2021_ConformalPrediction-TimeSeries__arXiv-2010.09107.pdf` | EnbPI — conformal intervals for time series. |
| `2025_ConformalPrediction-TimeSeries-ChangePoints__arXiv-2509.02844.pdf` | Conformal **under change points** — the exact failure mode our stress test induces. |
| `2022_CopulaConformalPrediction-MultiStepTimeSeries__arXiv-2212.03281.pdf` | Multi-step-ahead joint coverage — needed because mortality forecasts are h=1..20 horizons, not single-step. |

## G. Adjacent actuarial ML (related work / future paper #2 on explainability)

`RichmanWuthrich2021_LocalGLMnet__arXiv-2107.11059.pdf`,
`Brauer2023_Transformers-ActuarialNonLifePricing__arXiv-2311.07597.pdf`,
`2022_MultiTask-DiscriminationFree-InsurancePricing__arXiv-2207.02799.pdf`

## Still missing (paywalled, no legal open copy found)

| Paper | Where | Note |
|---|---|---|
| Perla, Richman, Scognamiglio & Wüthrich (2021), *Time-series forecasting of mortality rates using deep learning*, SAJ | doi:10.1080/03461238.2020.1867232 · SSRN 3595426 | SSRN blocks scripted download. Get via institutional login or SSRN in a browser. **Important benchmark** (shallow CNN generalization of LC). |
| Li & Lu (2017), *Coherent forecasting of mortality rates: a sparse VAR approach*, ASTIN | doi:10.1017/asb.2017.3 | Sparse-VAR comparator; we can reimplement from the paper description if access fails. |
| Gabrielli, Richman & Wüthrich (2020), *Neural network embedding of the ODP reserving model*, SAJ | doi:10.1080/03461238.2019.1633394 | Only needed for paper #2 (CAS reserving). |
| Tuljapurkar, Li & Boe (2000), *Universal pattern of mortality decline in the G7*, Nature 405:789 | doi:10.1038/35015561 | Motivates cross-population pooling. One-page citation, low priority. |
