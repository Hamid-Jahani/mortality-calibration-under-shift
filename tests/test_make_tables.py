"""Guards for scripts/make_tables.py on a tiny synthetic rows frame.

Ten rows: eight valid distributional / conformal cells, one CBD row, one
error row. The assertions are the scoring discipline the script exists to
enforce -- conformal proper scores never tabulated, conformal derived
quantities (e0 / e65 / annuity) never tabulated, CBD never ranked with
full-age families, error rows in no mean, every table produced, absent
regimes printed as an explicit placeholder, the second-pass GP family
printed as an explicit pending block until its parquet arrives, and
tab-populations generated from HMD bulk files rather than a parquet.
"""
import re
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_tables as mt  # noqa: E402

# distinctive sentinels: if any of them shows up where it must not, the
# discipline is broken
CONF_CRPS = 9.9999        # placeholder CRPS on the conformal row
ERR_COV95 = 0.0           # coverage_95 on the error row (valid LC/native = 0.90)
ERR_CRPS = 7.7777         # crps on the error row
CBD_CRPS = 0.0123         # CBD's CRPS, must appear only in its own block
CONF_E0 = 55.5555         # e0 quantities on the conformal row (filler samples)
GP_COV95 = 0.7777         # GP/native coverage, must appear only once GP rows are supplied


def _row(pop, sex, model, mech, **kw):
    base = dict(regime="shift", pop=pop, sex=sex, origin=2019, model=model, mechanism=mech,
                h=5, error=None, n_ages_scored=45 if model == "CBD" else 100, n_cells=500,
                n_zero_death_cells=3 if sex == "female" else 1,
                rmse_logmx=0.3, mae_logmx=0.2, crps_logmx=0.2, poisson_log_score=8.0,
                crps_counts=2.0, coverage_50=0.4, coverage_80=0.65, coverage_95=0.9,
                winkler_50=1.0, winkler_80=1.5, winkler_95=2.0, joint_path_coverage_95=0.5,
                pit_ks_stat=0.2, pit_ks_pvalue=0.01,
                murphy_reliability=0.01, murphy_resolution=0.0, murphy_uncertainty=0.05,
                murphy_brier=0.06, murphy_pit_reliability=0.02, murphy_pit_resolution=0.0,
                murphy_pit_uncertainty=0.08,
                e0_point=82.0, e0_q025=81.0, e0_q975=83.0, e0_obs=82.5, e0_error=-0.5,
                e65_point=20.0, e65_q025=19.0, e65_q975=21.0, e65_obs=22.0, e65_error=-2.0,
                ann65_point=16.0, ann65_q025=15.0, ann65_q975=17.0, ann65_obs=16.5, ann65_error=-0.5,
                scores_secondary=mech in mt.CONFORMAL_MECHANISMS, grid_secondary=False)
    for lvl in (50, 80, 95):
        for b in ("band0_24", "band25_64", "band65_99"):
            base[f"coverage_{lvl}_{b}"] = 0.9
    for k in range(1, 6):
        base.update({f"crps_h{k}": 0.2, f"logscore_h{k}": 8.0,
                     f"coverage95_h{k}": 0.9, f"winkler95_h{k}": 2.0})
    base.update(kw)
    return base


@pytest.fixture
def rows():
    r = [
        _row("SWE", "female", "LC", "native", coverage_95=0.90, crps_logmx=0.21, rmse_logmx=0.31),
        _row("SWE", "male", "LC", "native", coverage_95=0.90, crps_logmx=0.19, rmse_logmx=0.29),
        _row("SWE", "female", "PLC", "native", coverage_95=0.80, crps_logmx=0.25, rmse_logmx=0.20),
        _row("SWE", "male", "PLC", "native", coverage_95=0.80, crps_logmx=0.27, rmse_logmx=0.22),
        _row("SWE", "female", "LC", "pboot", coverage_95=0.85, crps_logmx=0.15, rmse_logmx=0.40),
        _row("SWE", "male", "LC", "pboot", coverage_95=0.85, crps_logmx=0.17, rmse_logmx=0.42),
        # conformal row: placeholder proper scores, 50/80 NaN by design, derived
        # quantities from uniform-in-interval filler samples
        _row("SWE", "female", "LC", "split_conf", crps_logmx=CONF_CRPS, poisson_log_score=CONF_CRPS,
             pit_ks_stat=CONF_CRPS, coverage_50=np.nan, coverage_80=np.nan, winkler_50=np.nan,
             winkler_80=np.nan, coverage_95=0.95,
             e0_point=CONF_E0, e0_q025=CONF_E0 - 1, e0_q975=CONF_E0 + 1, e0_obs=CONF_E0,
             e0_error=CONF_E0, e65_error=CONF_E0, ann65_error=CONF_E0),
        # CBD: 45 scored ages, e0 undefined
        _row("SWE", "female", "CBD", "native", crps_logmx=CBD_CRPS, rmse_logmx=0.05,
             e0_q025=np.nan, e0_q975=np.nan, e0_point=np.nan, e0_error=np.nan),
        _row("SWE", "male", "CBD", "native", crps_logmx=CBD_CRPS, rmse_logmx=0.05,
             e0_q025=np.nan, e0_q975=np.nan, e0_point=np.nan, e0_error=np.nan),
        # error row (structural design-floor): metrics deliberately poisoned
        _row("KOR", "female", "LC", "native", error="ValueError: inadmissible: n_train=8 < 15",
             coverage_95=ERR_COV95, crps_logmx=ERR_CRPS, rmse_logmx=ERR_CRPS,
             scores_secondary=None),
    ]
    return pd.DataFrame(r)


@pytest.fixture
def gp_rows():
    """The second-pass parquet: GP cells for the same regime / units."""
    return pd.DataFrame([
        _row("SWE", "female", "GP", "native", coverage_95=GP_COV95, crps_logmx=0.30, rmse_logmx=0.35),
        _row("SWE", "male", "GP", "native", coverage_95=GP_COV95, crps_logmx=0.32, rmse_logmx=0.37),
        _row("SWE", "female", "GP", "split_conf", crps_logmx=CONF_CRPS, coverage_50=np.nan,
             coverage_80=np.nan, coverage_95=0.93),
    ])


@pytest.fixture
def analysis():
    return {"shift": {
        "regimes": ["shift"], "loss": "crps", "alpha": 0.10, "n_rows": 10, "n_error_rows": 1,
        "mcs_classical_native_full_age": {
            "in_set": ["LC/native"], "eliminated": [["PLC/native", 0.02]],
            "p_values": {"PLC/native": 0.02, "LC/native": 1.0},
            "intersection": {"n_cells_kept": 2, "n_cells_dropped": 0, "loss": "crps"}},
        "mcs_classical_native": {"skipped": "arms do not share an age support: CBD/native=[45], LC/native=[100]. Restrict the arms."},
        "dm_native_vs_split": {"LC": {
            "mean_diff": 0.5, "se": 0.1, "t": 5.0, "p_value": 0.01, "n_clusters": 1,
            "intersection": {"n_cells_kept": 1, "n_cells_dropped": 1, "loss": "winkler95"}}},
    }}


#: the family x mechanism tables (one row per arm, one column block per regime)
FAMILY_MECH_TABLES = ("tab-h2-coverage", "tab-h3-joint", "tab-h4-age", "tab-h5-actuarial",
                      "tab-murphy", "tab-pit", "tab-twin-crises")


def _stacked(text):
    """Parse a stacked family x mechanism table into {(family, mech, regime): cells}.

    In the stacked layout the family and mechanism are printed only on the
    first regime row of each block, so a line-by-line match on the family name
    finds one row in three. This carries the labels forward.
    """
    out, fam, mech = {}, "", ""
    for ln in _body(text).splitlines():
        if "&" not in ln or ln.startswith("\\") or ln.startswith("%"):
            continue
        cells = [c.strip() for c in ln.rstrip("\ ").split("&")]
        if len(cells) < 3:
            continue
        if cells[0]:
            fam = cells[0]
        if cells[1]:
            mech = cells[1]
        out[(fam, mech, cells[2])] = cells
    return out


def _read(out, name):
    return (Path(out) / f"{name}.tex").read_text(encoding="utf-8")


def _body(text):
    """Table body without the notes: threeparttable notes start at
    \\begin{tablenotes}; longtable fragments carry theirs in a minipage
    after \\end{longtable}."""
    return text.split("\\begin{tablenotes}")[0].split("\\end{longtable}")[0]


# ---------------------------------------------------------------------------
# synthetic HMD bulk files for tab-populations
# ---------------------------------------------------------------------------

def _write_hmd(tmp_path, kind, cells):
    """cells: {(pop, year, age): (female, male)}; ages 0..110 ('110+'). A
    missing key -> the row is absent; value None -> '.' (HMD missing)."""
    lines = [f"{kind} (period 1x1), \tLast modified: 15 Jun 2026; Methods Protocol: v6 (2017)", "",
             "PopName    Year          Age             Female            Male           Total"]
    for (pop, year, age), v in sorted(cells.items()):
        a = "110+" if age == 110 else str(age)
        if v is None:
            lines.append(f"{pop:<10} {year:>5} {a:>10} {'.':>18} {'.':>15} {'.':>15}")
        else:
            f, m = v
            lines.append(f"{pop:<10} {year:>5} {a:>10} {f:>18.2f} {m:>15.2f} {f + m:>15.2f}")
    p = tmp_path / f"{kind}_1x1.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


@pytest.fixture
def hmd_files(tmp_path):
    """SWE 2000-2024 complete; KOR 2000-2024 but 2004 missing at every age
    (contiguity -> trains 2005-2019 = 15 years), one zero-exposure female cell
    in 2010, one missing male exposure in 2011 (both inside the training
    block), and zero-death test cells."""
    D, E = {}, {}
    for year in range(2000, 2025):
        for age in range(0, 111):
            D[("SWE", year, age)] = (10.0, 12.0)
            E[("SWE", year, age)] = (1000.0, 1000.0)
            if year == 2004:
                continue
            D[("KOR", year, age)] = (10.0, 12.0)
            E[("KOR", year, age)] = (1000.0, 1000.0)
    # zero-death test cells: SWE female 2020 ages 0-2 (D = 0.3 < 0.5); SWE male 2024 age 99
    for age in (0, 1, 2):
        D[("SWE", 2020, age)] = (0.3, 12.0)
    D[("SWE", 2024, 99)] = (10.0, 0.0)
    # zero-death cell above age 99 must not count
    D[("SWE", 2021, 105)] = (0.0, 0.0)
    # zero-exposure training cells for KOR: female E = 0 (2010, age 0); male missing (2011, age 5)
    E[("KOR", 2010, 0)] = (0.0, 1000.0)
    E[("KOR", 2011, 5)] = None
    # a zero-exposure cell outside the training block (KOR 2002, before the gap) must not count
    E[("KOR", 2002, 3)] = (0.0, 0.0)
    return _write_hmd(tmp_path, "Deaths", D), _write_hmd(tmp_path, "Exposures", E)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_all_tables_written_and_stamped(rows, analysis, tmp_path):
    written = mt.build_all(rows, analysis, None, tmp_path, sources=["_snap.parquet"], snapshot=True)
    names = sorted(p.name for p in written)
    assert names == sorted(f"{n}.tex" for n in mt.TABLE_NAMES)
    assert not (tmp_path / "tab-grid.tex").exists()          # static file never touched
    for p in written:
        text = p.read_text(encoding="utf-8")
        if p.stem == "tab-populations":
            # generated from the data files: never a snapshot, never stamped NOT FINAL
            assert text.startswith("% GENERATED FROM THE DATA FILES (not snapshot-derived)")
            assert "NOT FINAL" not in text
        else:
            assert text.startswith("% GENERATED SNAPSHOT - NOT FINAL - regenerate from results/")
            assert "_gp.parquet" in text.splitlines()[0]
        # page-spanning fragments (they carry their own caption/label):
        # the full infeasibility listing, H5 and the DM/MCS table overflowed
        # a float page and are emitted as longtables (measured 2026-08-28);
        # tab-infeasible and tab-h1-rankings joined them when the placebo
        # columns landed (measured 2026-08-31)
        # and the four family x mechanism tables joined when the stable control
        # landed: three regimes of column blocks ran up to 118 pt past the text
        # block, so they carry the regimes as ROWS and span pages instead
        # (measured 2026-09-04)
        if p.stem in ("tab-infeasible-full", "tab-h5-actuarial", "tab-dm-mcs",
                      "tab-infeasible", "tab-h1-rankings", "tab-h2-coverage",
                      "tab-h3-joint", "tab-pit", "tab-murphy", "tab-h4-age"):
            assert "\\begin{longtable}" in text and "\\end{longtable}" in text
            assert "\\begin{tabular}" not in text
        else:
            assert "\\begin{tabular}" in text and "\\bottomrule" in text
            assert "\\begin{tablenotes}" in text


def test_float_sizes(rows, analysis, tmp_path):
    """Every threeparttable fragment: \\scriptsize body, tight column spacing,
    notes in \\scriptsize.

    The guard is against a fragment drifting BACK to loose settings (\\small,
    3-4 pt columns), which is what overflowed the text block. A table is
    allowed to go TIGHTER than the 2.5 pt default -- tab-murphy carries
    fourteen-plus columns and is emitted at 2 pt for exactly that reason --
    so the spacing is checked as a bound, not as a literal.
    """
    mt.build_all(rows, analysis, None, tmp_path)
    for name in mt.TABLE_NAMES:
        text = _read(tmp_path, name)
        # page-spanning longtable fragments: three regimes of column blocks
        # overflowed the text block horizontally, so these carry the regimes as
        # ROWS and span pages, with their own caption and label
        if any(k in str(name) for k in ("tab-infeasible-full", "tab-h5-actuarial",
                                        "tab-dm-mcs", "tab-infeasible", "tab-h1-rankings",
                                        "tab-h2-coverage", "tab-h3-joint",
                                        "tab-pit", "tab-murphy", "tab-h4-age")):
            # page-spanning longtable fragments size themselves (own caption/label)
            assert "\\begin{longtable}" in text, name
            continue
        assert "\\scriptsize\\renewcommand{\\arraystretch}{0.90}\\setlength{\\tabcolsep}{" in text, name
        sep = re.search(r"\\setlength\{\\tabcolsep\}\{([0-9.]+)pt\}", text)
        assert sep and float(sep.group(1)) <= 2.5, f"{name}: tabcolsep {sep and sep.group(1)}"
        if name == "tab-infeasible-full":
            assert "\\begin{minipage}{\\linewidth}\\scriptsize" in text
        else:
            assert "\\begin{tablenotes}\\scriptsize" in text, name
        assert "\\small" not in text and "tabcolsep}{3pt}" not in text and "tabcolsep}{4pt}" not in text


def test_conformal_scores_never_ranked(rows, analysis, tmp_path):
    mt.build_all(rows, analysis, None, tmp_path)
    h1 = _read(tmp_path, "tab-h1-rankings")
    assert "split" not in _body(h1)     # no conformal row in the body
    assert f"{CONF_CRPS:.3f}" not in h1
    pit = _read(tmp_path, "tab-pit")
    assert "split" not in _body(pit)
    assert f"{CONF_CRPS:.3f}" not in pit
    # conformal rows ARE in the coverage table, with 50/80 shown as n/a
    # stacked layout: one row per regime, so take the shift row (the one with data)
    h2 = _read(tmp_path, "tab-h2-coverage")
    conf_line = " & ".join(_stacked(h2)[("Lee--Carter (SVD)", "split", "shift")])
    assert conf_line.count("n/a") == 2 and "0.950" in conf_line
    # murphy: hit-based columns present, PIT reliability n/a on the conformal row
    mu = _read(tmp_path, "tab-murphy")
    cells = _stacked(mu)[("Lee--Carter (SVD)", "split", "shift")]
    assert cells[3:7] == ["0.0100", "0.0000", "0.0500", "n/a"]


def test_h5_conformal_rows_are_na(rows, analysis, tmp_path):
    """Defect 1: conformal e0/e65/ann65 quantiles come from uniform-in-interval
    filler samples -- printed n/a, never tabulated, with the tablenote."""
    mt.build_all(rows, analysis, None, tmp_path)
    h5 = _read(tmp_path, "tab-h5-actuarial")
    lines = h5.splitlines()
    i = next(j for j, ln in enumerate(lines) if "& split & stable &" in ln)
    conf_line = lines[i + 1]   # stacked layout: shift row follows the arm's stable row
    cells = [c.strip() for c in conf_line.split("&")]
    assert cells[2] == "shift"
    assert cells[3:9] == ["n/a"] * 6                      # six derived-quantity columns
    assert cells[9] == "1" and cells[10].replace("\\\\", "").strip() == "0"  # n / n_err
    assert f"{CONF_E0:.2f}" not in h5 and f"{CONF_E0:.3f}" not in h5
    assert "1.000" not in conf_line                      # cov share never computed for it
    assert ("derived-quantity intervals require predictive samples; conformal mechanisms "
            "yield interval bounds on $\\log m_x$ only (addendum 2 \\S3)") in h5
    assert "conformal rows are included and unflagged" not in h5
    # the stats helper itself never computes them (not merely a formatting mask)
    st = mt._h5_stats(mt.valid_rows(mt.prepare_rows(rows)), rows.iloc[0:0], [])
    conf = st[st["mechanism"] == "split_conf"].iloc[0]
    assert np.isnan(conf["e0_cov"]) and np.isnan(conf["e0_err"]) and conf["e0_n"] == 0


def test_cbd_separate_block(rows, analysis, tmp_path):
    mt.build_all(rows, analysis, None, tmp_path)
    h1 = _read(tmp_path, "tab-h1-rankings")
    body = _body(h1)
    main_block, cbd_block = body.split("CBD (M5), ages 55--99")
    assert "CBD" not in main_block.split("full-age families")[1]
    assert f"{CBD_CRPS:.3f}" not in main_block
    assert f"{CBD_CRPS:.3f}" in cbd_block
    # ranks in the main block run over the three full-age arms only
    ranks = [int(ln.split("&")[3]) for ln in main_block.splitlines()
             if ln.startswith(("Lee--Carter", "Poisson"))]
    assert sorted(ranks) == [1, 2, 3]


def test_error_rows_enter_no_mean(rows, analysis, tmp_path):
    mt.build_all(rows, analysis, None, tmp_path)
    for name in ("tab-h2-coverage", "tab-h3-joint", "tab-h4-age", "tab-h1-rankings",
                 "tab-twin-crises", "tab-h5-actuarial", "tab-murphy", "tab-pit"):
        assert f"{ERR_CRPS:.3f}" not in _read(tmp_path, name)
    # LC/native: two valid rows at cov95 = 0.90 plus one error row at 0.0. The
    # mean must be 0.900 (0.600 with the error row in) and n / n_err = 2 / 1.
    h2 = _read(tmp_path, "tab-h2-coverage")
    cells = _stacked(h2)[("Lee--Carter (SVD)", "native", "shift")]
    assert cells[5] == "0.900" and cells[7] == "2" and cells[8] == "1"   # cov95, n, n_err
    # tab-h3-joint is stacked (regime column); tab-twin-crises keeps regime blocks
    j = _stacked(_read(tmp_path, "tab-h3-joint"))[("Lee--Carter (SVD)", "native", "shift")]
    assert j[3] == "0.900"
    line = [ln for ln in _read(tmp_path, "tab-twin-crises").splitlines()
            if ln.startswith("Lee--Carter (SVD) & native")][0]
    assert [c.strip() for c in line.split("&")][3] == "0.900"
    h1 = _read(tmp_path, "tab-h1-rankings")
    lc_h1 = [ln for ln in h1.splitlines() if ln.startswith("Lee--Carter (SVD) & native")][0]
    assert [c.strip() for c in lc_h1.split("&")][2] == "0.300"          # mean(0.31, 0.29), not with 7.7777
    # the error row is tabulated, classified structural, KOR named: compact in
    # the main-text fragment, per population in the appendix longtable
    inf = _read(tmp_path, "tab-infeasible")
    assert "Lee--Carter (SVD) & native & structural & KOR & 1 & 1" in inf
    full = _read(tmp_path, "tab-infeasible-full")
    assert "KOR & Lee--Carter (SVD) & native & structural" in full
    for text in (inf, full):
        assert "machine" not in [ln.split("&")[3].strip() for ln in text.splitlines() if ln.count("&") == 5]


def test_infeasible_compact_and_full(rows, analysis, tmp_path):
    """Defect 5: main-text fragment aggregates to (family, mechanism, class)
    with the populations affected; the per-population listing is a separate
    longtable fragment."""
    more = rows.copy()
    extra = [_row("CHL", "male", "LC", "native", error="ValueError: inadmissible: n_train=9 < 15"),
             _row("CHL", "female", "LC", "native", error="ValueError: inadmissible: n_train=9 < 15"),
             _row("CHL", "female", "SVAR", "native",
                  error="ValueError: 953/1000 coefficient draws remain explosive after 100 redraws")]
    more = pd.concat([more, pd.DataFrame(extra)], ignore_index=True)
    mt.build_all(more, analysis, None, tmp_path)
    inf = _read(tmp_path, "tab-infeasible")
    body = _body(inf)
    assert "Lee--Carter (SVD) & native & structural & CHL, KOR & 2 & 3" in body
    assert "sparse VAR & native & method & CHL & 1 & 1" in body
    assert "Populations affected" in body and "explosive" not in body   # no messages in the compact table
    assert "tab:infeasible-full" in inf
    full = _read(tmp_path, "tab-infeasible-full")
    assert full.count("\\begin{longtable}") == 1 and "\\endfirsthead" in full and "\\endlastfoot" in full
    assert "\\caption{" in full and "\\label{tab:infeasible-full}" in full
    assert "CHL & Lee--Carter (SVD) & native & structural" in full
    assert "CHL & sparse VAR & native & method" in full and "k/N coefficient draws" in full
    assert "KOR & Lee--Carter (SVD) & native & structural" in full
    assert "\\begin{threeparttable}" not in full


def test_machine_failure_aborts(rows, analysis, tmp_path):
    bad = rows.copy()
    bad.loc[bad.index[-1], "error"] = "MemoryError: Unable to allocate 3 GiB"
    with pytest.raises(SystemExit, match="machine-failure"):
        mt.build_all(bad, analysis, None, tmp_path)
    assert not list(tmp_path.glob("*.tex"))


def test_absent_regimes_are_explicit_placeholders(rows, analysis, tmp_path):
    mt.build_all(rows, analysis, None, tmp_path)
    for name in mt.TABLE_NAMES:
        if name == "tab-populations":
            continue
        text = _read(tmp_path, name)
        assert "pending" in text, name
    tc = _read(tmp_path, "tab-twin-crises")
    assert "placebo" in tc and "\\emph{pending}" in tc
    dm = _read(tmp_path, "tab-dm-mcs")
    assert "winkler95" in dm and "crps" in dm and "skipped:" in dm
    # no HMD bulk files -> every data column of tab-populations is pending
    pops = _read(tmp_path, "tab-populations")
    assert "SWE & " + " & ".join(["\\emph{pending}"] * 7) + " & yes & neutral" in pops
    assert "GBR\\_SCO" in pops and "civilian-only" in pops


def test_gp_block_pending_until_second_pass(rows, analysis, tmp_path):
    """Defect 2: every family x mechanism table carries an explicit GP block;
    without GP rows it says pending (second-pass parquet)."""
    mt.build_all(rows, analysis, None, tmp_path)
    for name in FAMILY_MECH_TABLES + ("tab-h1-rankings", "tab-infeasible", "tab-infeasible-full"):
        text = _read(tmp_path, name)
        assert "GP: \\emph{pending} (second-pass parquet)" in _body(text), name
        assert "multi-output GP" in text, name
        assert f"{GP_COV95:.3f}" not in text, name
    # the block sits inside the tabular at the family's grid position, spanning the regime's columns
    h2 = _read(tmp_path, "tab-h2-coverage")
    gp_rows_ = [c for k, c in _stacked(h2).items() if k[0] == "multi-output GP"]
    assert gp_rows_, "no GP block in the stacked table"
    span = "\\multicolumn{6}{l}{GP: \\emph{pending} (second-pass parquet)}"
    # only the regime that HAS rows shows the GP pending span; the other two
    # regimes are absent from the fixture and show the regime pending span
    shift_row = [c for k, c in _stacked(h2).items()
                 if k[0] == "multi-output GP" and k[2] == "shift"]
    assert shift_row and span in " & ".join(shift_row[0])
    tc = _read(tmp_path, "tab-twin-crises")
    gp_line = [ln for ln in tc.splitlines() if ln.startswith("multi-output GP")][0]
    assert "\\multicolumn{3}{l}{GP: \\emph{pending} (second-pass parquet)}" in gp_line


def test_gp_rows_merge_by_regime(rows, gp_rows, analysis, tmp_path):
    """The second-pass parquet's rows join the family x mechanism tables and
    the ranking; the pending block disappears for that regime."""
    pq = tmp_path / "shift.parquet"
    gp = tmp_path / "shift_gp.parquet"
    rows.to_parquet(pq)
    gp_rows.to_parquet(gp)
    df = mt.load_rows([str(pq), str(gp)])
    assert (df["model"] == "GP").sum() == 3 and set(df["_source"]) == {"shift.parquet", "shift_gp.parquet"}
    mt.build_all(df, analysis, None, tmp_path)
    for name in FAMILY_MECH_TABLES + ("tab-h1-rankings",):
        text = _read(tmp_path, name)
        assert "second-pass parquet" not in _body(text), name
        assert "multi-output GP & native" in text, name
    h2 = _read(tmp_path, "tab-h2-coverage")
    st = _stacked(h2)
    cells = st[("multi-output GP", "native", "shift")]
    assert cells[5] == f"{GP_COV95:.3f}" and cells[7] == "2"
    # both families' split arms are present as their own stacked blocks
    assert ("Lee--Carter (SVD)", "split", "shift") in st
    assert ("multi-output GP", "split", "shift") in st
    h1 = _read(tmp_path, "tab-h1-rankings")
    ranks = [int(ln.split("&")[3]) for ln in _body(h1).splitlines()
             if ln.startswith(("Lee--Carter", "Poisson", "multi-output GP"))]
    assert sorted(ranks) == [1, 2, 3, 4]           # GP/native ranked with the full-age arms
    # a cell present in both parquets would be averaged twice: refused
    with pytest.raises(SystemExit, match="duplicate cell rows"):
        mt.load_rows([str(pq), str(pq)])


def test_twin_crises_has_no_family_restriction_note(rows, analysis, tmp_path):
    """Defect 3: PREREGISTRATION places no family restriction on the placebo."""
    mt.build_all(rows, analysis, None, tmp_path)
    tc = _read(tmp_path, "tab-twin-crises")
    assert "Classical families only" not in tc
    assert "no registered placebo arm" not in tc
    assert "no family restriction" in tc and "no transfer regression" in tc


def test_h5_coverage_share_and_cbd_e0_undefined(rows, analysis, tmp_path):
    mt.build_all(rows, analysis, None, tmp_path)
    h5 = _read(tmp_path, "tab-h5-actuarial")
    lines = h5.splitlines()
    # stacked layout: the arm's first row is the (pending) stable regime;
    # its shift row follows on the next line with blank family/mech cells
    i = next(j for j, ln in enumerate(lines)
             if ln.startswith("Lee--Carter (SVD) & native"))
    cells = [c.strip() for c in lines[i + 1].split("&")]
    assert cells[2] == "shift"
    # e0 obs 82.5 inside [81, 83] -> 1.000; e65 obs 22 outside [19, 21] -> 0.000
    assert cells[3] == "1.000" and cells[5] == "0.000"
    i = next(j for j, ln in enumerate(lines) if ln.startswith("CBD (M5) & native"))
    assert lines[i + 1].split("&")[3].strip() == "--"


def test_populations_from_data_files(rows, analysis, hmd_files, tmp_path):
    """Defect 4: tab-populations is generated from the HMD bulk files, never
    from a parquet -- first / last year, contiguous training years, zero-E
    training cells by sex, zero-D test cells by sex."""
    deaths, exposures = hmd_files
    facts = mt.population_facts(deaths, exposures)
    assert facts["SWE"] == {"first_year": 2000, "last_year": 2024, "n_train": 20,
                            "zero_E_f": 0, "zero_E_m": 0, "zero_D_f": 3, "zero_D_m": 1}
    # KOR: 2004 missing -> contiguous block 2005-2019 (15 years); the 2002
    # zero-exposure cell precedes the gap and is outside the training panel.
    # Female: E = 0 (2010) + the '.' row (2011); male: the '.' row only.
    assert facts["KOR"] == {"first_year": 2000, "last_year": 2024, "n_train": 15,
                            "zero_E_f": 2, "zero_E_m": 1, "zero_D_f": 0, "zero_D_m": 0}
    out = tmp_path / "tables"
    mt.build_all(rows, analysis, None, out, sources=["_snap.parquet"], snapshot=True,
                 hmd_deaths=deaths, hmd_exposures=exposures)
    pops = _read(out, "tab-populations")
    assert pops.startswith("% GENERATED FROM THE DATA FILES (not snapshot-derived)")
    assert "NOT FINAL" not in pops and "_snap.parquet" not in pops
    assert "SWE & 2000 & 2024 & 20 & 0 & 0 & 3 & 1 & yes & neutral" in pops
    assert "KOR & 2000 & 2024 & 15 & 2 & 1 & 0 & 0 & no & none" in pops
    # populations absent from the (synthetic) files stay pending; nothing from the parquet leaks in
    assert "USA & " + " & ".join(["\\emph{pending}"] * 7) + " & no & none" in pops
    assert "Generated from the data files" in pops and "MANIFEST.sha256" in pops
    assert "cell rows" not in pops                       # no parquet bookkeeping note
    assert "Zero-$E$ train" in pops and "Zero-$D$ test" in pops


def test_sensitivities_strata_columns(rows, analysis, tmp_path):
    # the scripts/sensitivities.py contract (its module docstring)
    sens = {"contract_version": 1, "snapshot": True, "strata": {"placebo": {
        "pooled": {"_meta": {}, "LC/native": {"coverage_95": 0.70, "joint_path_coverage_95": 0.4, "n_cells": 22}},
        "neutral": {"_meta": {}, "LC/native": {"coverage_95": 0.71, "joint_path_coverage_95": 0.4, "n_cells": 14}},
        "belligerent_total": {"_meta": {}},
        "civilian_only": {"_meta": {}, "LC/native": {"coverage_95": None, "joint_path_coverage_95": None, "n_cells": 0}},
    }}}
    # a pending block (placebo parquet absent upstream) must fall back to the placeholder
    mt.build_all(rows, analysis, {"strata": {"placebo": "pending"}}, tmp_path)
    assert "\\emph{pending}" in _read(tmp_path, "tab-twin-crises")
    mt.build_all(rows, analysis, sens, tmp_path)
    tc = _read(tmp_path, "tab-twin-crises")
    assert "by stratum" in tc and "neutral" in tc and "0.710" in tc
    assert "sensitivities JSON" in tc


def test_cli_smoke(rows, analysis, tmp_path, capsys):
    pq = tmp_path / "_tiny_snapshot.parquet"
    rows.to_parquet(pq)
    aj = tmp_path / "_tiny_analysis.json"
    aj.write_text(json.dumps(analysis["shift"]), encoding="utf-8")
    out = tmp_path / "tables"
    missing = tmp_path / "no_such_file.txt"
    argv = ["--parquet", str(pq), "--analysis", str(aj), "--out", str(out),
            "--hmd-deaths", str(missing), "--hmd-exposures", str(missing)]
    assert mt.main(argv) == 0
    assert len(list(out.glob("tab-*.tex"))) == len(mt.TABLE_NAMES)
    assert "pending second-pass parquet" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        mt.main(argv + ["--final"])
    # defect 7: the --sensitivities help names the key the table reads
    with pytest.raises(SystemExit):
        mt.main(["--help"])
    assert "strata.placebo" in capsys.readouterr().out
