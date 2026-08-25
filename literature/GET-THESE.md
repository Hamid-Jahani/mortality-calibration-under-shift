# Papers you need to fetch manually — direct links

Everything programmatically obtainable has been obtained. These nine resist scripted download. Ranked by how much they block the work.

## 🔴 Blocking — read before the novelty claim is finalised

**1. Dowd, Cairns, Blake, Coughlan, Epstein & Khalaf-Allah (2010)** — *Backtesting Stochastic Mortality Models: An Ex-Post Evaluation of Multi-Period-Ahead Density Forecasts*, NAAJ 14(3):281–298
- SSRN (free in browser): **https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1396201**
- DOI: https://doi.org/10.1080/10920277.2010.10597592
- *Why it blocks:* this is direct prior art for the audit half. Density-forecast backtesting for mortality was done in 2010. We need to know exactly what it covers before claiming anything is new.

**2. Perla, Richman, Scognamiglio & Wüthrich (2021)** — *Time-series forecasting of mortality rates using deep learning*, SAJ 2021(7):572–598
- SSRN (free in browser): **https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3595426**
- DOI: https://doi.org/10.1080/03461238.2020.1867232
- *Why:* the shallow-CNN LC baseline. Required model in the roster.

**3. Schnürch & Korn (2022)** — *Point and Interval Forecasts of Death Rates Using Neural Networks*, ASTIN Bulletin 52(1)
- SSRN (free in browser): **https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3796051**
- DOI: https://doi.org/10.1017/asb.2021.34
- *Why:* the only neural mortality paper explicitly about *interval* forecasts. Miyata cites it as producing seed-driven intervals — central to the "what counts as uncertainty" argument.

## 🟠 Needed — each defines a baseline model or a test

**4. Renshaw & Haberman (2006)** — *A cohort-based extension to the Lee–Carter model for mortality reduction factors*, IME 38(3):556–570
- DOI: https://doi.org/10.1016/j.insmatheco.2005.12.001
- Try the City, University of London repository (authors were at Cass): https://openaccess.city.ac.uk/
- *Why:* the APC/cohort baseline.

**5. Dowd, Cairns, Blake, Coughlan, Epstein & Khalaf-Allah (2010)** — *Evaluating the Goodness of Fit of Stochastic Mortality Models*, IME 47:255–265
- DOI: https://doi.org/10.1016/j.insmatheco.2010.06.006
- Cairns notes on his page: *"Please e-mail me for a reprint"* — A.Cairns@ma.hw.ac.uk
- *Why:* companion to #1; the fit-diagnostic half of the same programme.

**6. Li & Lu (2017)** — *Coherent forecasting of mortality rates: a sparse vector-autoregression approach*, ASTIN Bulletin 47(2):563–600
- DOI: https://doi.org/10.1017/asb.2017.15
- *Why:* the sparse-VAR statistical comparator. Reimplementable from the paper if access fails.

**7. Hansen, Lunde & Nason (2011)** — *The Model Confidence Set*, Econometrica 79(2):453–497
- DOI: https://doi.org/10.3982/ECTA5771
- *Why:* the multiple-comparison procedure for the model ranking. Note we already have the arXiv paper applying MCS to age-specific mortality, which may be enough to cite.

**8. Cairns, Blake, Kessler & Kessler (2020)** — *The Impact of COVID-19 on Future Higher-Age Mortality*
- Listed on https://www.macs.hw.ac.uk/~andrewc/papers/ but the link there resolves to a different paper. E-mail the author, or check the Pensions Institute archive.
- *Why:* the accelerated-deaths framing for pandemic years; Barigou et al. build on it.

## ⚪ Nice to have

**9. Tuljapurkar, Li & Boe (2000)** — *A universal pattern of mortality decline in the G7 countries*, Nature 405:789–792
- DOI: https://doi.org/10.1038/35015561
- One motivating sentence only. Cite from the abstract if needed.

---

## How to get the SSRN ones (#1, #2, #3)

Open the link, click **Download This Paper**. They are free — SSRN simply refuses non-browser requests. Save into `literature/pdf/` and tell me; I will re-run the text extraction and fold them into the analysis.

## If you have institutional access

Cambridge Core covers #3, #6. Elsevier/ScienceDirect covers #4, #5. Taylor & Francis covers #1, #2. Wiley covers #7. One library session gets all of them.
