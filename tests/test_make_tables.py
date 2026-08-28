"""Guards for scripts/make_tables.py on a tiny synthetic rows frame.

Ten rows: eight valid distributional / conformal cells, one CBD row, one
error row. The assertions are the scoring discipline the script exists to
enforce -- conformal proper scores never tabulated, CBD never ranked with
full-age families, error rows in no mean, every table produced, absent
regimes printed as an explicit placeholder.
"""
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
        # conformal row: placeholder proper scores, 50/80 NaN by design
        _row("SWE", "female", "LC", "split_conf", crps_logmx=CONF_CRPS, poisson_log_score=CONF_CRPS,
             pit_ks_stat=CONF_CRPS, coverage_50=np.nan, coverage_80=np.nan, winkler_50=np.nan,
             winkler_80=np.nan, coverage_95=0.95),
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


def _read(out, name):
    return (Path(out) / f"{name}.tex").read_text(encoding="utf-8")


def test_all_tables_written_and_stamped(rows, analysis, tmp_path):
    written = mt.build_all(rows, analysis, None, tmp_path, sources=["_snap.parquet"], snapshot=True)
    names = sorted(p.name for p in written)
    assert names == sorted(f"{n}.tex" for n in mt.TABLE_NAMES)
    assert not (tmp_path / "tab-grid.tex").exists()          # static file never touched
    for p in written:
        text = p.read_text(encoding="utf-8")
        assert text.startswith("% GENERATED SNAPSHOT - NOT FINAL - regenerate from results/")
        assert "\\begin{tabular}" in text and "\\bottomrule" in text
        assert "\\begin{tablenotes}" in text


def test_conformal_scores_never_ranked(rows, analysis, tmp_path):
    mt.build_all(rows, analysis, None, tmp_path)
    h1 = _read(tmp_path, "tab-h1-rankings")
    assert "split" not in h1.split("\\begin{tablenotes}")[0]     # no conformal row in the body
    assert f"{CONF_CRPS:.3f}" not in h1
    pit = _read(tmp_path, "tab-pit")
    assert "split" not in pit.split("\\begin{tablenotes}")[0]
    assert f"{CONF_CRPS:.3f}" not in pit
    # conformal rows ARE in the coverage table, with 50/80 shown as n/a
    h2 = _read(tmp_path, "tab-h2-coverage")
    conf_line = [ln for ln in h2.splitlines() if "& split &" in ln][0]
    assert conf_line.count("n/a") == 2 and "0.950" in conf_line
    # murphy: hit-based columns present, PIT reliability n/a on the conformal row
    mu = _read(tmp_path, "tab-murphy")
    conf_line = [ln for ln in mu.splitlines() if "& split &" in ln][0]
    cells = [c.strip() for c in conf_line.split("&")]
    assert cells[3:7] == ["0.0100", "0.0000", "0.0500", "n/a"]


def test_cbd_separate_block(rows, analysis, tmp_path):
    mt.build_all(rows, analysis, None, tmp_path)
    h1 = _read(tmp_path, "tab-h1-rankings")
    body = h1.split("\\begin{tablenotes}")[0]
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
    lc_native = [ln for ln in h2.splitlines() if ln.startswith("Lee--Carter (SVD) & native")][0]
    cells = [c.strip() for c in lc_native.split("&")]
    assert cells[5] == "0.900" and cells[7] == "2" and cells[8] == "1"   # cov95, n, n_err
    for name, col in (("tab-h3-joint", 3), ("tab-twin-crises", 3)):   # [fam, mech, stable/placebo pending, cov95]
        line = [ln for ln in _read(tmp_path, name).splitlines()
                if ln.startswith("Lee--Carter (SVD) & native")][0]
        assert [c.strip() for c in line.split("&")][col] == "0.900", name
    h1 = _read(tmp_path, "tab-h1-rankings")
    lc_h1 = [ln for ln in h1.splitlines() if ln.startswith("Lee--Carter (SVD) & native")][0]
    assert [c.strip() for c in lc_h1.split("&")][2] == "0.300"          # mean(0.31, 0.29), not with 7.7777
    # the error row is tabulated, classified structural, KOR named
    inf = _read(tmp_path, "tab-infeasible")
    assert "KOR & Lee--Carter (SVD) & native & structural" in inf
    assert "machine" not in [ln.split("&")[3].strip() for ln in inf.splitlines() if ln.count("&") == 5]


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
    pops = _read(tmp_path, "tab-populations")
    assert "SWE & \\emph{pending} & \\emph{pending} & 3 & 1 & yes & neutral" in pops
    assert "GBR\\_SCO" in pops and "civilian-only" in pops


def test_h5_coverage_share_and_cbd_e0_undefined(rows, analysis, tmp_path):
    mt.build_all(rows, analysis, None, tmp_path)
    h5 = _read(tmp_path, "tab-h5-actuarial")
    lc = [ln for ln in h5.splitlines() if ln.startswith("Lee--Carter (SVD) & native")][0]
    cells = [c.strip() for c in lc.split("&")]
    # e0 obs 82.5 inside [81, 83] -> 1.000; e65 obs 22 outside [19, 21] -> 0.000
    assert cells[3] == "1.000" and cells[5] == "0.000"
    cbd = [ln for ln in h5.splitlines() if ln.startswith("CBD (M5) & native")][0]
    assert cbd.split("&")[3].strip() == "--"


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


def test_cli_smoke(rows, analysis, tmp_path):
    pq = tmp_path / "_tiny_snapshot.parquet"
    rows.to_parquet(pq)
    aj = tmp_path / "_tiny_analysis.json"
    aj.write_text(json.dumps(analysis["shift"]), encoding="utf-8")
    out = tmp_path / "tables"
    assert mt.main(["--parquet", str(pq), "--analysis", str(aj), "--out", str(out)]) == 0
    assert len(list(out.glob("tab-*.tex"))) == len(mt.TABLE_NAMES)
    with pytest.raises(SystemExit):
        mt.main(["--parquet", str(pq), "--analysis", str(aj), "--out", str(out), "--final"])
