# Manual-fetch queue — CLOSED 2026-08-26

All nine papers that resisted scripted download have been obtained, filed in
`literature/pdf/`, and text-extracted into `literature/txt/`. Corpus is now
**59 PDFs / 52 extracted texts**. Nothing is outstanding.

Kept as a record of provenance and version caveats, which matter for citation.

## Obtained

| # | Paper | File stem | Version held |
|---|---|---|---|
| 1 | Dowd, Cairns, Blake et al. (2010), *Backtesting Stochastic Mortality Models*, NAAJ 14(3):281–298 | `DowdCairnsBlakeEtAl2010_Backtesting-…_NAAJ` (published) and `…_NAAJ-PI0803` (working paper) | ✅ **published, 18 pp, obtained 2026-09-03**; the PI-0803 preprint (43 pp) is kept for provenance only |
| 2 | Perla, Richman, Scognamiglio & Wüthrich (2021), *Time-Series Forecasting … Deep Learning*, SAJ 2021(7):572–598 | `PerlaRichmanScognamiglioWuthrich2021_…_SAJ` | ⚠️ preprint "Version of May 6, 2020", 26 pp |
| 3 | Schnürch & Korn (2022), *Point and Interval Forecasts of Death Rates Using Neural Networks*, ASTIN 52(1) | `SchnurchKorn2022_…_ASTIN` | ✅ published, 28 pp |
| 4 | Renshaw & Haberman (2006), *A cohort-based extension to the Lee–Carter model*, IME 38(3):556–570 | `RenshawHaberman2006_…_IME` | ✅ published, 15 pp |
| 5 | Dowd, Cairns, Blake et al. (2010), *Evaluating the Goodness of Fit of Stochastic Mortality Models*, IME 47:255–265 | `DowdCairnsBlakeEtAl2010_EvaluatingGoodnessOfFit-…_IME` | ✅ published, 11 pp |
| 6 | Li & Lu (2017), *Coherent Forecasting … Sparse VAR*, ASTIN 47(2):563–600 | `LiLu2017_…_ASTIN` | ✅ published, 38 pp |
| 7 | Hansen, Lunde & Nason (2011), *The Model Confidence Set*, Econometrica 79(2):453–497 | `HansenLundeNason2011_…_Econometrica` | ✅ published, 45 pp |
| 8 | Cairns, Blake, Kessler & Kessler (2020), *The Impact of COVID-19 on Future Higher-Age Mortality* | `CairnsBlakeKesslerKessler2020_…_PI-2007` | ⚠️ Pensions Institute DP **PI-2007**, 34 pp |
| 9 | Tuljapurkar, Li & Boe (2000), *A Universal Pattern of Mortality Decline in the G7*, Nature 405:789–792 | `TuljapurkarLiBoe2000_…_Nature` | ✅ published, 4 pp |

**Cite the published version for #2 and #8; the file on disk is a working
paper.** Verify any quoted number or section reference against the published
text — preprints and final articles diverge.

**This caveat has already bitten once.** #1's published text was obtained
2026-09-03 and the working paper diverges materially: its summary bullet
"Forecast performance tends to improve with higher ages" is absent from the
published article, which plots two ages and reports a bias comparison
instead, and the dependence footnote is numbered 9 in the working paper but
10 in the published article. `02-related-work.tex` had quoted the
working-paper sentence against the published citation. Both are corrected;
see `docs/PRIOR-ART.md`.

## Verified DOIs

```
1  10.1080/10920277.2010.10597592     6  10.1017/asb.2016.37
2  10.1080/03461238.2020.1867232      7  10.3982/ECTA5771
3  10.1017/asb.2021.34                8  10.2139/ssrn.3606988
4  10.1016/j.insmatheco.2005.12.001   9  10.1038/35015561
5  10.1016/j.insmatheco.2010.06.006
```

All confirmed to resolve against Crossref on 2026-08-26. Note #6: an earlier
revision of this file listed `10.1017/asb.2017.15`, which resolves to a
*different* ASTIN paper ("Bayesian Analysis of Big Data in Insurance Predictive
Modeling"). `paper/references.bib` was always correct.

## Prior-art scan — READ BEFORE THE NOVELTY CLAIM IS WRITTEN

Lexical scan of the extracted text, 2026-08-26. **These are term counts, not a
reading.** Treat as a map of where to look, not as evidence.

Both closest prior-art papers *do* evaluate interval performance, contrary to a
first impression from the word "coverage" alone:

- **Schnürch & Korn (2022)** explicitly measures coverage — "prediction interval
  coverage probability (PICP)", and observes that some methods "ignore some
  uncertainty as their realized coverage probabilities are significantly [below
  nominal]". This is direct prior art for **H2** (marginal under-coverage of
  neural mortality intervals). "prediction interval" ×31, "calibrat*" ×13.
- **Dowd et al. (2010, backtesting)** evaluates intervals through
  **exceedances** (×29) — counts of outcomes falling above/below risk bounds
  against expected proportions. That is coverage assessment in different
  vocabulary. "backtest" ×47. A footnote says PIT backtests could "in
  principle" also be performed, implying they were not.

What neither paper appears to contain, across all six prior-art texts scanned
(Schnürch–Korn, both Dowd 2010s, Cairns 2020, Li–Lu, Renshaw–Haberman):

```
CRPS                 0      scoring rule         0      log score        0
joint coverage       0      conformal            0      distribution shift 0
structural break     0      empirical coverage   0      nominal          0
```

Provisional reading — **must be confirmed by actually reading #1 and #3**:
H2 has genuine prior art and should be framed as replication-under-shift rather
than discovery. The joint/path-coverage hypothesis (**H3**), the crossed
model × UQ-mechanism design, proper scoring rules, and the conformal arms remain
unaddressed in this literature.
