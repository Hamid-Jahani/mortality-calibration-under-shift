# Papers I could NOT download

Every paper cited in `Datasets and Paper Pathways for Research.docx` that is not in `pdf/` or `pdf-other-datasets/`, with the exact reason and the fastest route to get it.

**Blocking mechanism legend** — `SSRN-403`: SSRN serves the abstract page but returns 403 to any scripted PDF fetch. `PAYWALL`: publisher subscription, no legal open copy located. `NO-OA`: no open-access or author-hosted copy found after search.

---

## Priority 1 — needed for THIS paper (HMD / mortality)

| # | Paper | Identifier | Blocked by | How to get it |
|---|---|---|---|---|
| 1 | **Perla, Richman, Scognamiglio & Wüthrich (2021)**, *Time-series forecasting of mortality rates using deep learning*, Scandinavian Actuarial Journal 2021(7):572–598 | `10.1080/03461238.2020.1867232` · SSRN `3595426` | SSRN-403 + T&F PAYWALL | Open https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3595426 **in a browser** and click Download — it is free there, just not scriptable. **Do this one.** It is the shallow-CNN LC generalization and a required benchmark. |
| 2 | **Li, H. & Lu, Y. (2017)**, *Coherent forecasting of mortality rates: a sparse vector-autoregression approach*, ASTIN Bulletin 47(2):563–600 | `10.1017/asb.2017.15` | Cambridge PAYWALL (not OA) | Institutional login, or reimplement from the published description — the model is a penalised VAR on log-rates and is fully specified in the abstract-level literature. |
| 3 | **Tuljapurkar, Li & Boe (2000)**, *A universal pattern of mortality decline in the G7 countries*, Nature 405:789–792 | `10.1038/35015561` | Nature PAYWALL | Low priority — used for one motivating sentence about cross-population commonality. Cite from abstract if needed. |

## Priority 2 — HRS track (paper #3 candidate)

| # | Paper | Blocked by |
|---|---|---|
| 4 | Li, J. S.-H., Shao, A. W. & Sherris, M. (2017), multistate latent-factor mortality/disability modelling on HRS | NO-OA — the doc cites only an HRS bibliography entry, journal not pinned down |
| 5 | Weiss et al. (2025), transformer-based mortality prediction on HRS (38,193 adults, 126 risk factors) | NO-OA — HRS bibliography entry #19950 only |
| 6 | Badolato et al. (2026), *The Limits of Predicting Individual-Level Longevity*, Demography | PAYWALL (Duke UP) |

*Got instead:* both Gustman NBER pension/wealth working papers (free from NBER).

## Priority 3 — MEPS track

| # | Paper | Blocked by |
|---|---|---|
| 7 | Duncan, Loginov & Ludkovski (2016), *Testing Alternative Regression Frameworks for Predictive Modeling of Health Care Costs*, NAAJ | T&F PAYWALL |
| 8 | McClellan et al. (2023), MEPS ML predictive-mean-matching study, Health Services Research | Wiley PAYWALL |
| 9 | Yue & Hong (2012), Bayesian Tobit quantile regression on MEPS | SAGE PAYWALL |
| 10 | Smith, West & Zhang (2021), marginalized two-part survey models on MEPS, HSR | Wiley PAYWALL |

## Priority 4 — freMTPL2 / pricing track

| # | Paper | Blocked by |
|---|---|---|
| 11 | Noll, Salzmann & Wüthrich (2020), *Case Study: French Motor Third-Party Liability Claims* | SSRN-403 (`3164764`) — free in a browser |
| 12 | Lindholm, Richman, Tsanakas & Wüthrich (2021), *Discrimination-Free Insurance Pricing*, ASTIN Bulletin | Cambridge PAYWALL |

*Got instead:* LocalGLMnet, Brauer transformers, and two adjacent arXiv discrimination-free papers.

## Priority 5 — CAS reserving track (paper #2 candidate)

| # | Paper | Blocked by |
|---|---|---|
| 13 | Gabrielli, Richman & Wüthrich (2020), *Neural Network Embedding of the Over-Dispersed Poisson Reserving Model*, SAJ | SSRN-403 + T&F PAYWALL — **worth grabbing in a browser** if paper #2 goes ahead |
| 14 | Maciak, Mizera & Pešta (2022), *Functional Profile Techniques for Claims Reserving*, ASTIN | Cambridge PAYWALL |
| 15 | Gao & Meng (2017), *Stochastic Claims Reserving via a Bayesian Spline Model*, ASTIN | Cambridge PAYWALL |
| 16 | Wüthrich (2018), *Machine Learning in Individual Claims Reserving*, SAJ | T&F PAYWALL |

*Got instead:* **Mack (1993)** distribution-free chain ladder, free from the CAS archive.

## Priority 6 — Fannie Mae / credit track

| # | Paper | Blocked by |
|---|---|---|
| 17 | Deng, Quigley & Van Order (2000), *Mortgage Terminations, Heterogeneity and the Exercise of Mortgage Options*, Econometrica | Wiley PAYWALL |
| 18 | Ciochetti, Deng, Gao & Yao (2002), Real Estate Economics | Wiley PAYWALL |

*Got instead:* DeepHit (AAAI, free) and DeepSurv (arXiv) — the competing-risks ML benchmarks.

## Priority 7 — VCDB / cyber track

| # | Paper | Blocked by |
|---|---|---|
| 19 | Bessy-Roland, Boumezoued & Hillairet (2021), *Multivariate Hawkes Process for Cyber Insurance*, AAS 15(1):14–39 | Cambridge PAYWALL — the "PDF" endpoint returns an HTML consent page, not the article |
| 20 | Sun, Xu & Zhao (2021), *Modeling Malicious Hacking Data Breach Risks*, NAAJ | T&F PAYWALL |

*Got instead:* Neural Hawkes (Mei & Eisner), large-scale multivariate Hawkes (Lemonnier), doubly-censored Hawkes (Xu/Luo/Zha), plus two cyber-insurance Hawkes arXiv papers.

## Priority 8 — ORX / operational risk track (needs membership anyway)

| # | Paper | Blocked by |
|---|---|---|
| 21 | Shevchenko (2010), *Implementing Loss Distribution Approach for Operational Risk*, ASMB | Wiley PAYWALL |
| 22 | Kelliher et al. (2016), *Good Practice Guide to Setting Inputs for Operational Risk Models*, BAJ | Cambridge PAYWALL |
| 23 | Afonso & Corte Real (2016), *Using Weighted Distributions to Model Operational Risk*, ASTIN | Cambridge PAYWALL |
| 24 | Chavez-Demoulin, Embrechts & Hofert (2016), *An Extreme Value Approach for Modeling Operational Risk Losses Depending on Covariates*, JRI | Wiley PAYWALL |
| 25 | Feuerverger (2016), *On Goodness of Fit for Operational Risk*, Int. Statistical Review | Wiley PAYWALL |
| 26 | Abdymomunov & Ergen (2017), *Tail Dependence and Systemic Risk in Operational Losses*, Int. Review of Finance | Wiley PAYWALL |

---

## The short version

**Only one missing paper actually blocks this project: #1, Perla et al. (2021).** It is free on SSRN in a browser — SSRN just refuses scripted downloads. Two minutes of your time.

Everything else on this list is either (a) a different dataset track, or (b) replaceable by an open substitute already downloaded.

If you have institutional access (Cambridge Core + Taylor & Francis covers #1, #2, #12, #13, #14, #15, #16, #19), say so and I will list exact DOIs to pull in one session.
