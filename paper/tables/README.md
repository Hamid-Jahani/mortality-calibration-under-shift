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

## `--variant main`: the abridged fragments for the venue-fitted manuscript

`docs/SPLIT-SPEC.md` rule 4. Each of the five hypothesis tables costs 3 typeset
pages on its own and the venue-fitted manuscript has room for about 5 pages of
exhibits in total, so it needs one-page views — **generated**, never
hand-trimmed. Same script, same parquet inputs, same provenance header and the
same snapshot stamp; only how much of the grid is printed changes:

    python scripts/make_tables.py --parquet ... --analysis ... \
        --sensitivities results/sensitivities.json --variant main --final
    # --out defaults to paper/submission/tables for this variant

Written: `tab-h1-rankings-main`, `tab-h2-coverage-main`, `tab-h3-joint-main`,
`tab-h4-age-main`, `tab-h5-actuarial-main` — and nothing else. The default
(`--variant full`) is untouched and still writes exactly the twelve
`TABLE_NAMES` fragments into `paper/tables/`, which is what `paper/main.tex`
builds from.

The abridgement doctrine, stated once so the manuscript text and the generator
agree: **the twenty own-law arms in full, the thirty conformal arms as a
per-mechanism envelope, stable + shift only, no per-cell counts.** Per table:

| fragment | body | dropped |
|---|---|---|
| `tab-h1-rankings-main` | shift regime only, 18 full-age arms + the CBD block | the stable and placebo blocks; their rank correlations are in the note |
| `tab-h2-coverage-main` | 20 own-law arms, 3 + 1 envelope rows, stable and shift side by side, plus a generated signed `Δ|cov−0.95|` | the placebo, the 50/80 levels, `n` / `n_err`, the 30 conformal arms individually |
| `tab-h3-joint-main` | as H2, plus the generated independence benchmark `c^H` at each arm's own marginal rate and `joint − c^H` | the placebo (`H=9`), `n` / `n_err`, the 30 conformal arms individually |
| `tab-h4-age-main` | shift bands 0–24 / 25–64 / 65–99 plus the gradient `Δ = cov(65–99) − cov(25–64)` for shift **and** for the stable control | the stable and placebo band triples, `n` / `n_err`, the 30 conformal arms individually |
| `tab-h5-actuarial-main` | the 20 own-law arms only, `e65` and annuity, stable and shift | the placebo, the `e0` pair, `n` / `n_err`, and the 30 conformal rows (every one of their cells is `n/a` by construction, so no number is lost) |

Rules the abridged path enforces:

- **Envelope rows are a summary, not a cell.** `mean [min, max]` over the
  full-age families carrying one wrapper; both endpoints are named in the note
  so each resolves to a row of the supplementary table. `RESTRICTED_AGE_FAMILIES`
  is the authority: CBD (45 scored ages) never enters a full-age envelope and
  keeps its own sub-block and its own conformal row.
- **Thin arms carry a dagger** with their row count in the note, because the
  `n` / `n_err` columns are gone (sparse VAR native is scored on 297 of 520
  stable units, its bootstrap on 238).
- **Caption and label live inside the fragment** (`TableWriter.write_float`,
  unlike `write`), so an abridged exhibit cannot be typeset without saying that
  it is abridged and naming the supplementary table that carries the full grid.
  Input them at top level, **not** inside a `table` float:
  `\inputtable{tab-h2-coverage-main}`.
- **Supplement S-numbers are generated**, from `SUPPLEMENT_TABLE_ORDER` in
  `scripts/make_tables.py`: `paper/submission/supp/S3-full-tables.tex` must
  input the unabridged fragments in that order (`tab-populations` = Table S1,
  … `tab-infeasible-full` = Table S12). Reorder there and the constant moves
  with it; a test pins the mapping.
- **New quantities.** `Δ|cov−0.95|` (H2), `c^H` and `joint − c^H` (H3) and the
  age gradients (H4) are computed only in this path and appear in no
  supplementary table. `c^H` is defined as the cell-mean coverage raised to the
  regime's horizon count `H` (the `h` column: 5 in stable and shift, 9 in the
  placebo) — the definition behind the counts Section 6 quotes.

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
