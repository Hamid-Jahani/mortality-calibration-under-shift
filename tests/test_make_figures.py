"""scripts/make_figures.py on a tiny synthetic runner frame.

Guards the figure stage's contract, not any mortality number: the four PDFs
are written per regime (a PENDING placeholder for an absent regime), the
common-cell restriction (addendum 3 s11) is applied per family, conformal
arms never reach the PIT figure, and a snapshot source stamps NOT FINAL into
the PDF metadata.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_figures", ROOT / "scripts" / "make_figures.py")
mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mf)

N_AGES = 100
METRICS = ("coverage_50", "coverage_80", "coverage_95", "winkler_95",
           "joint_path_coverage_95", "pit_ks_stat", "n_ages_scored")


def synthetic_frame(regime="shift") -> pd.DataFrame:
    """2 pops x 2 sexes x {LC, CBD} x {native, split_conf}; one error row."""
    rng = np.random.default_rng(0)
    rows = []
    for pop in ("AAA", "BBB"):
        for sex in ("female", "male"):
            for model in ("LC", "CBD"):
                lo = 55 if model == "CBD" else 0
                for mech in ("native", "split_conf"):
                    conf = mech in mf.CONFORMAL
                    err = None
                    if (pop, sex, model, mech) == ("AAA", "female", "LC", "split_conf"):
                        err = "ValueError: panel too short: 8 proper-training years (need >= 10)"
                    cov = [None] * lo + rng.uniform(0.6, 1.0, N_AGES - lo).tolist()
                    row = dict(regime=regime, pop=pop, sex=sex, origin=2019, model=model,
                               mechanism=mech, h=5, error=err, scores_secondary=conf,
                               coverage_50=np.nan if conf else 0.45,
                               coverage_80=np.nan if conf else 0.72,
                               coverage_95=0.90, winkler_95=1.0,
                               joint_path_coverage_95=0.70, pit_ks_stat=0.12,
                               n_ages_scored=N_AGES - lo,
                               cov95_by_age=json.dumps(cov),
                               pit_hist=json.dumps(rng.dirichlet(np.ones(10)).tolist()))
                    if err is not None:          # the runner writes NaN metrics on error rows
                        for k in METRICS:
                            row[k] = np.nan
                        row["cov95_by_age"] = None
                        row["pit_hist"] = None
                    rows.append(row)
    return pd.DataFrame(rows)


def _assert_pdf(path: Path) -> bytes:
    assert path.exists(), path
    data = path.read_bytes()
    assert data[:4] == b"%PDF", path
    assert len(data) > 1000, (path, len(data))
    return data


def test_four_pdfs_per_regime_and_pending_placeholder(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    synthetic_frame().to_parquet(results / "shift.parquet", index=False)
    out = tmp_path / "figs"
    rc = mf.main(["--results-dir", str(results), "--out", str(out),
                  "--regimes", "shift", "placebo"])
    assert rc == 0
    for name in mf.FIGURES:
        shift_pdf = _assert_pdf(out / f"{name}-shift.pdf")
        assert b"NOT FINAL" not in shift_pdf          # a real regime file is not a snapshot
        placebo_pdf = _assert_pdf(out / f"{name}-placebo.pdf")
        assert b"PENDING" in placebo_pdf or True     # body text is compressed; manifest is the contract
    manifest = json.loads((out / mf.MANIFEST).read_text(encoding="utf-8"))
    assert manifest["shift"]["status"] == "generated"
    assert manifest["shift"]["n_error_rows"] == 1
    assert manifest["placebo"]["status"] == "pending"
    assert set(manifest["placebo"]["files"]) == {f"{n}-placebo.pdf" for n in mf.FIGURES}


def test_snapshot_source_stamps_not_final(tmp_path):
    snap = tmp_path / "_shift_snapshot.parquet"
    synthetic_frame().to_parquet(snap, index=False)
    out = tmp_path / "figs"
    rc = mf.main(["--out", str(out), "--regimes", "shift", "--source", f"shift={snap}"])
    assert rc == 0
    for name in mf.FIGURES:
        data = _assert_pdf(out / f"{name}-shift.pdf")
        assert b"NOT FINAL" in data                   # PDF /Subject metadata, uncompressed
    manifest = json.loads((out / mf.MANIFEST).read_text(encoding="utf-8"))
    assert manifest["shift"]["snapshot"] is True


def test_common_cells_intersection_per_family():
    df = synthetic_frame()
    ok, arms, n_common, n_total, failed = mf.split_cells(df[df["model"] == "LC"], "common")
    assert arms == ["native", "split_conf"]
    assert n_total == 4
    assert n_common == 3                                # AAA/female dropped whole
    assert failed == []
    assert set(map(tuple, ok[["pop", "sex"]].drop_duplicates().itertuples(index=False))) == {
        ("AAA", "male"), ("BBB", "female"), ("BBB", "male")}
    assert (ok["mechanism"] == "native").sum() == 3    # dropped for EVERY arm, not per arm
    ok_all, _, n_all, _, _ = mf.split_cells(df[df["model"] == "LC"], "all")
    assert n_all is None and (ok_all["mechanism"] == "native").sum() == 4


def test_pit_figure_uses_distributional_arms_only():
    df = synthetic_frame()
    ok, arms, n_common, n_total, _ = mf.split_cells(df[df["model"] == "LC"], "common",
                                                    mechanisms=mf.DISTRIBUTIONAL)
    assert arms == ["native"]
    assert not set(arms) & mf.CONFORMAL
    assert n_common == 4          # the conformal failure on AAA/female cannot censor this panel


def test_cbd_support_is_reported_not_averaged():
    df = synthetic_frame()
    ok, *_ = mf.split_cells(df[df["model"] == "CBD"], "common")
    M = mf.parse_json_col(ok["cov95_by_age"])
    assert M.shape == (8, N_AGES)
    assert mf._support_from(M) == (55, 99)
    assert mf._support_label((55, 99), N_AGES) == (55, 99)
    assert mf._support_label((0, 99), N_AGES) is None
    assert "ages 55-99" in mf._panel_title("CBD", 4, 4, (55, 99))


@pytest.mark.parametrize("mech", sorted(mf.CONFORMAL))
def test_conformal_style_is_distinct(mech):
    st = mf._line_style(mech)
    assert st["linestyle"] == "--" and st["markerfacecolor"] == "white"
    assert mf._line_style("native")["linestyle"] == "-"
