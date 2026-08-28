# paper/tables

Every table in the paper is **generated**, never typed. The producer is
`scripts/make_tables.py`, which reads the per-cell parquet outputs of
`src/mortcal/runner.py` — `results/stable.parquet`, `results/shift.parquet`,
`results/placebo.parquet` plus the second-pass GP files
`results/<regime>_gp.parquet` (`scripts/launch_sweeps.sh` runs the
multi-output GP family separately; its rows merge by regime) — plus the
`scripts/analyse.py` JSON per regime and (optionally)
`results/sensitivities.json`, and writes one `booktabs` fragment (wrapped in
`threeparttable` with a `tablenotes` block) per table into this directory:

    python scripts/make_tables.py --parquet results/shift.parquet \
        --parquet results/shift_gp.parquet --parquet results/placebo.parquet \
        --analysis results/shift_analysis.json \
        --sensitivities results/sensitivities.json --out paper/tables

`--hmd-deaths` / `--hmd-exposures` default to the HMD bulk files under
`Dataset/` (pinned in `data/MANIFEST.sha256`); they feed `tab-populations`
only.

Conventions the producer enforces:

- **Float sizes.** Every threeparttable fragment is `\footnotesize` with
  `\tabcolsep` 2.5pt and its `tablenotes` in `\scriptsize`, so the wide
  family x mechanism x regime tables fit a text-width float.
- **Snapshot stamp.** Fragments built from a snapshot input (basename starting
  with `_`) carry the first-line comment `% GENERATED SNAPSHOT - NOT FINAL -
  regenerate from results/<regime>.parquet + results/<regime>_gp.parquet`.
  `tab-populations` is the exception: it is generated **from the data files**
  (HMD `Deaths_1x1` + `Exposures_1x1`), never from a parquet, and is stamped
  `% GENERATED FROM THE DATA FILES (not snapshot-derived)` instead.
- **Pending placeholders.** Absent regimes appear as explicit *pending* rows.
  The GP family always has an explicit block in every family x mechanism
  table; until the second-pass parquet is supplied for a regime that block
  reads `GP: pending (second-pass parquet)`.
- **Conformal rows** (split, EnbPI, copula) are compared on `coverage_95` /
  `winkler_95` only: their 50/80 columns are n/a, their CRPS / log score /
  PIT are never tabulated, and their `e0` / `e65` / annuity rows in
  `tab-h5-actuarial` are n/a (the runner's derived-quantity quantiles for
  those rows come from uniform-in-interval filler samples, not a predictive
  distribution; addendum 2 §3).
- A machine-failure error row (`scripts/final_qa.py` class) aborts
  generation. Duplicate cells across the supplied parquets abort too.

`tab-conformal-secondary` and `tab-robustness` are not yet produced.
`paper/sections/*.tex` pulls fragments in with `\inputtable{<name>}`; a
missing fragment is rendered as a red `\todo{}` placeholder by the
`\inputtable` macro in `main.tex`, so the draft always compiles.

Intended filenames (hooked already from the section files):

| File | Section | Content |
|---|---|---|
| `tab-grid.tex` | Design | admissible family × mechanism grid (static; transcribed from `docs/GRID.md`) |
| `tab-populations.tex` | Data | per shift population, from the HMD bulk files: first / last year of single-age data, contiguous training years to 2019, zero-exposure training cells (E = 0 or missing, ages 0–99) by sex, zero-death test cells (D < 0.5, 2020–24) by sex, placebo eligibility and addendum-1 stratum |
| `tab-h1-rankings.tex` | Results H1 | RMSE rank vs CRPS rank per family, shift regime |
| `tab-h2-coverage.tex` | Results H2 | empirical 50/80/95 coverage and Winkler by family × mechanism × regime |
| `tab-h3-joint.tex` | Results H3 | marginal vs joint path coverage, h = 1…5, per family |
| `tab-h4-age.tex` | Results H4 | coverage by age band (0–24, 25–64, 65–99), shift regime |
| `tab-pit.tex` | Results | PIT decile frequencies and uniformity statistics |
| `tab-murphy.tex` | Results | Murphy calibration / resolution split of the proper scores |
| `tab-dm-mcs.tex` | Results | Diebold–Mariano t (wild cluster bootstrap p) and 90% MCS membership |
| `tab-twin-crises.tex` | Results | 1914–22 vs 2020–24 side-by-side coverage, pooled and by placebo stratum (neutral / belligerent / civilian-only, `PREREGISTRATION-ADDENDUM-1.md`); all families, GP from the second pass |
| `tab-h5-actuarial.tex` | Actuarial impact H5 | coverage and error of e0, e65, ä65 @ 2% intervals (distributional rows; conformal rows n/a) |
| `tab-infeasible.tex` | Results | **compact**: design-floor (structural) and method-failure cells aggregated to family × mechanism × class with the populations affected and the row count |
| `tab-infeasible-full.tex` | Appendix | the per-(population, family, mechanism) listing with error excerpts, as a `longtable` fragment: needs `\usepackage{longtable}` and must be `\input` **outside** any `table` float (it carries its own `\caption` / `\label{tab:infeasible-full}`) |
| `tab-conformal-secondary.tex` | Appendix | flagged proper scores for conformal cells |
| `tab-robustness.tex` | Appendix | sensitivities: drop-2024, age-cap-90, drop USA+CHL, register-vs-census split; placebo drop-GBR_SCO, neutral-only, DNK without 1921–22 |

Do not hand-edit generated fragments; fix the producer.
