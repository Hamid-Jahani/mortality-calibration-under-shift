"""Parser tests against the real local HMD files (skipped if data absent)."""
from pathlib import Path

import pytest

DS = Path(__file__).resolve().parents[1] / "Dataset"
DEATHS = DS / "deaths" / "Deaths_1x1" / "Deaths_1x1.txt"
EXPOS = DS / "exposures" / "Exposures_1x1" / "Exposures_1x1.txt"
MX = DS / "death_rates" / "Mx_1x1" / "Mx_1x1.txt"

pytestmark = pytest.mark.skipif(not DEATHS.exists(), reason="HMD data not present")


def test_read_bulk_first_row_matches_raw_file():
    from mortcal.data import read_bulk_1x1

    df = read_bulk_1x1(DEATHS, pops=["AUS"])
    row = df[(df.pop_codes if False else df["pop"].eq("AUS")) & df["year"].eq(1921) & df["age"].eq(0)].iloc[0]
    assert row["female"] == pytest.approx(3842.31)
    assert row["male"] == pytest.approx(5124.54)
    assert row["total"] == pytest.approx(8966.85)


def test_open_age_group_becomes_110():
    from mortcal.data import read_bulk_1x1

    df = read_bulk_1x1(DEATHS, pops=["AUS"])
    assert df["age"].max() == 110
    assert df["age"].min() == 0


def test_panel_mx_identity_against_hmd_mx_file():
    """D/E computed by us must match HMD's own Mx to published precision."""
    from mortcal.data import build_panel, read_bulk_1x1

    panel = build_panel(DEATHS, EXPOS, pops=["USA"])
    ours = panel[(panel["pop"] == "USA") & (panel["year"] == 2019)
                 & (panel["age"] == 65) & (panel["sex"] == "male")]["mx"].iloc[0]
    mx = read_bulk_1x1(MX, pops=["USA"])
    theirs = mx[mx["year"].eq(2019) & mx["age"].eq(65)]["male"].iloc[0]
    assert ours == pytest.approx(theirs, rel=2e-3)


def test_panel_drops_bad_exposures_and_caps_age():
    from mortcal.data import build_panel

    panel = build_panel(DEATHS, EXPOS, pops=["AUS"])
    assert panel["age"].max() == 99
    assert (panel["E"] > 0).all()
    assert panel["mx"].notna().all()
