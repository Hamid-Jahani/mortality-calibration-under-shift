# Spec: venue-fitted manuscript + online supplement

Status: measurements final, exhibit ledger pending the section survey.
The 110-page paper in `paper/` is **frozen and untouched**. Everything here lands in
`paper/submission/`, which shares `paper/tables/`, `paper/figures/` and `paper/references.bib`
by relative path so no generated exhibit is ever duplicated in source.

## Why a split is required

`Annals of Actuarial Science`: "Submitted articles should be no more than 35 pages", excess to
supplementary material. `ASTIN Bulletin`: over 30 pages, "consider splitting into shorter
contributions". Both host supplements online, **not typeset or copyedited**. Sources and dates
in `paper/submission/COMPLIANCE.md`.

## Measured budget (not estimated)

Built `paper/main.tex` with `\inputtable` and `\inputfigure` redefined to no-ops:

| quantity | measured |
|---|---|
| full build | 110 pp |
| prose + references only, no exhibits | **61 pp** |
| therefore exhibits | **49 pp** |
| prose words across the nine sections | **36,317** |
| prose density | ~640 words/page |

Prose-only page cost by section (from the no-exhibit build):

| section | pp | words |
|---|---|---|
| 1 Introduction | 5 | 3,542 |
| 2 Related work | 6 | 2,727 |
| 3 Data | 2 | 2,607 |
| 4 Study design | 8 | 4,742 |
| 5 Metrics | 4 | 3,092 |
| 6 Results | 12 | 8,250 |
| 7 Actuarial impact | 4 | 3,323 |
| 8 Discussion | 9 | 6,573 |
| 9 Conclusion | 2 | 1,338 |
| References (88 entries) | 6 | — |
| Appendices A + B prose | 3 | 123 |

Exhibit page cost, each table compiled alone:

| exhibit | pp | | exhibit | pp |
|---|---|---|---|---|
| tab-grid | 1 | | tab-pit | 2 |
| tab-h1-rankings | 2 | | tab-murphy | 3 |
| tab-h2-coverage | 3 | | tab-dm-mcs | 3 |
| tab-h3-joint | 3 | | tab-twin-crises | 1 |
| tab-h4-age | 3 | | tab-populations | 1 |
| tab-h5-actuarial | 3 | | tab-infeasible | 4 |
| each of 12 figures | 1 (full-page float) | | tab-infeasible-full | 22 |

**Consequence.** The five hypothesis tables alone are 15 pages. There is no arrangement of
whole exhibits that fits 35 pages. The main manuscript needs both (a) abridged exhibits and
(b) a ~60% prose reduction. This is a rewrite, not a re-arrangement — the honest scope.

## Target composition of the main manuscript (<= 34 pp)

| block | pp |
|---|---|
| title, author, abstract (199 words), keywords | 1.5 |
| prose, all nine sections | 21 |
| exhibits: 4 abridged tables + grid + 1 figure | 5 |
| required statements | 0.5 |
| references | 6 |

Per-section prose word targets, from 36,317 down to ~13,700:

| section | now | target |
|---|---|---|
| 1 Introduction | 3,542 | 1,600 |
| 2 Related work | 2,727 | 1,000 |
| 3 Data | 2,607 | 1,000 |
| 4 Study design | 4,742 | 1,700 |
| 5 Metrics | 3,092 | 1,100 |
| 6 Results | 8,250 | 3,600 |
| 7 Actuarial impact | 3,323 | 1,400 |
| 8 Discussion | 6,573 | 1,700 |
| 9 Conclusion | 1,338 | 600 |

## Rules that bound the compression

1. **No registered outcome may live only in the supplement.** Every pre-registered hypothesis
   keeps, in the main body: its verdict, its headline numbers, and an explicit pointer to the
   supplementary table carrying the full per-cell evidence. This paper audits others for
   selective reporting; it cannot practise it.
2. **Every deviation from pre-registration stays in the main text.** The H2 refutation, the H3
   re-specification and the addendum-4 GP window cap are findings, not housekeeping.
3. **Limitations stay in the main text.** Compress the wording, never the content.
4. **Abridged exhibits are generated, never hand-trimmed.** `scripts/make_tables.py` gains a
   variant that writes compact fragments into `paper/submission/tables/`; the provenance
   header and the parquet inputs are unchanged. A hand-edited number in a table would break
   the reproducibility claim this paper rests on.
5. **The supplement is self-contained and S-numbered** (Table S1, Figure S1, Section S1), and
   is submitted as a separate PDF supplied exactly as it will appear online.

## Supplement contents

Everything moved out, in this order: full methods detail dropped from Sections 3-5; the
complete family x mechanism x regime tables (h1, h2, h3, h4, h5, murphy, pit, dm-mcs,
twin-crises, populations); all 12 figures; Appendix A (conformal secondary scores, robustness
slices); Appendix B (the 22-page ledger of cells that produced no valid row); validation-gate
detail; and the sensitivity slices.

## Build

`paper/submission/manuscript.tex` and `paper/submission/supplement.tex`, both via latexmk,
sharing the parent tree's assets. `paper/main.tex` keeps building the 110-page version
unchanged.
