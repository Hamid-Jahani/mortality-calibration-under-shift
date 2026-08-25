# paper/tables

Every table in the paper is **generated**, never typed. The intended producer
is `scripts/analyze.py` (not yet written), which reads the per-cell parquet
outputs of `src/mortcal/runner.py` — `results/stable.parquet`,
`results/shift.parquet`, `results/placebo.parquet` — and writes one
`booktabs` fragment per table into this directory. `paper/sections/*.tex`
pulls them in with `\input{tables/<name>}`; a missing fragment is rendered
as a red `\todo{}` placeholder by the `\inputtable` macro in `main.tex`, so
the draft always compiles.

Intended filenames (hooked already from the section files):

| File | Section | Content |
|---|---|---|
| `tab-grid.tex` | Design | admissible family × mechanism grid (static; transcribed from `docs/GRID.md`) |
| `tab-populations.tex` | Data | population codes, start year, last year, cells with E = 0 |
| `tab-h1-rankings.tex` | Results H1 | RMSE rank vs CRPS rank per family, shift regime |
| `tab-h2-coverage.tex` | Results H2 | empirical 50/80/95 coverage and Winkler by family × mechanism × regime |
| `tab-h3-joint.tex` | Results H3 | marginal vs joint path coverage, h = 1…5, per family |
| `tab-h4-age.tex` | Results H4 | coverage by age band (0–24, 25–64, 65–99), shift regime |
| `tab-pit.tex` | Results | PIT decile frequencies and uniformity statistics |
| `tab-murphy.tex` | Results | Murphy calibration / resolution split of the proper scores |
| `tab-dm-mcs.tex` | Results | Diebold–Mariano t (wild cluster bootstrap p) and 90% MCS membership |
| `tab-twin-crises.tex` | Results | 1914–22 vs 2020–24 side-by-side coverage, pooled and by placebo stratum (neutral / belligerent / civilian-only, `PREREGISTRATION-ADDENDUM-1.md`) |
| `tab-h5-actuarial.tex` | Actuarial impact H5 | coverage and error of e0, e65, ä65 @ 2% intervals |
| `tab-conformal-secondary.tex` | Appendix | flagged proper scores for conformal cells |
| `tab-robustness.tex` | Appendix | sensitivities: drop-2024, age-cap-90, drop USA+CHL, register-vs-census split; placebo drop-GBR_SCO, neutral-only, DNK without 1921–22 |

Do not hand-edit generated fragments; fix the producer.
