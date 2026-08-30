"""Guards for scripts/h5_extras.py.

Synthetic-frame checks of the two computations (annuity tail-shortfall
shares, H1 rank-correlation cluster bootstrap) plus key/shape checks on the
committed results/h5_extras.json when it exists. The assertions are the
guards the script promises: error rows and scores_secondary rows in no
share, shares in [0, 1], conditional mean shortfall computed over exactly
the exceeding cells, and the bootstrap deterministic given the seed.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import h5_extras  # noqa: E402
import make_tables as mt  # noqa: E402

POPS = ["AUT", "BEL", "CHE", "DNK", "FIN", "NOR"]
ARMS = [("LC", "native"), ("PLC", "native"), ("RH", "native"), ("NB", "native")]


def _rows():
    rng = np.random.default_rng(7)
    rows = []
    for m, u in ARMS:
        for p in POPS:
            for sex in ("female", "male"):
                # obs above q975 on AUT female only for LC -> known share
                exceed_hi = (m == "LC" and p == "AUT" and sex == "female")
                exceed_lo = (m == "RH" and p in ("BEL", "CHE"))
                obs = 17.5 if exceed_hi else (14.6 if exceed_lo else 16.2)
                rows.append(dict(
                    regime="shift", pop=p, sex=sex, origin=2019,
                    model=m, mechanism=u, error=None,
                    scores_secondary=False, grid_secondary=False,
                    rmse_logmx=float(rng.uniform(0.1, 0.5)),
                    crps_logmx=float(rng.uniform(0.1, 0.5)),
                    poisson_log_score=float(rng.uniform(5, 9)),
                    ann65_point=16.0, ann65_q025=15.0, ann65_q975=17.0,
                    ann65_obs=obs, ann65_error=16.0 - obs))
    # an error row and a conformal row: neither may enter any share
    rows.append(dict(regime="shift", pop="AUT", sex="male", origin=2019,
                     model="LC", mechanism="pboot", error="fit diverged",
                     scores_secondary=False, grid_secondary=False,
                     rmse_logmx=np.nan, crps_logmx=np.nan,
                     poisson_log_score=np.nan, ann65_point=np.nan,
                     ann65_q025=np.nan, ann65_q975=np.nan,
                     ann65_obs=np.nan, ann65_error=np.nan))
    rows.append(dict(regime="shift", pop="AUT", sex="female", origin=2019,
                     model="LC", mechanism="split_conf", error=None,
                     scores_secondary=True, grid_secondary=False,
                     rmse_logmx=0.2, crps_logmx=9.9, poisson_log_score=9.9,
                     ann65_point=16.0, ann65_q025=15.9, ann65_q975=16.1,
                     ann65_obs=17.0, ann65_error=-1.0))
    return mt.prepare_rows(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def df():
    return _rows()


def test_shortfall_shares_and_exclusions(df):
    arms = h5_extras.annuity_shortfall(df)
    by = {(r["model"], r["mechanism"]): r for r in arms}
    # conformal and error arms never tabulated as distributional cells
    assert ("LC", "split_conf") not in by
    assert ("LC", "pboot") not in by
    for r in arms:
        for side in ("upper", "lower"):
            s = r[f"{side}_exceed_share"]
            assert s is not None and 0.0 <= s <= 1.0
            ms = r[f"{side}_mean_rel_shortfall"]
            assert (ms is None) == (r[f"{side}_exceed_n"] == 0)
            if ms is not None:
                assert ms > 0.0
    # known cell: LC/native exceeds q975 on 1 of 12 units by 0.5/16
    lc = by[("LC", "native")]
    assert lc["n_admissible"] == 12
    assert lc["upper_exceed_n"] == 1
    assert lc["upper_exceed_share"] == pytest.approx(1 / 12)
    assert lc["upper_mean_rel_shortfall"] == pytest.approx(0.5 / 16.0)
    # RH/native: 4 of 12 units below q025 by 0.4/16
    rh = by[("RH", "native")]
    assert rh["lower_exceed_n"] == 4
    assert rh["lower_exceed_share"] == pytest.approx(4 / 12)
    assert rh["lower_mean_rel_shortfall"] == pytest.approx(0.4 / 16.0)


def test_bootstrap_deterministic_given_seed(df):
    a = h5_extras.h1_bootstrap(df, B=64, seed=123)
    b = h5_extras.h1_bootstrap(df, B=64, seed=123)
    assert a == b
    c = h5_extras.h1_bootstrap(df, B=64, seed=124)
    assert (a["rho_rmse_crps"]["ci95"] != c["rho_rmse_crps"]["ci95"]
            or a["rho_rmse_logscore"]["ci95"] != c["rho_rmse_logscore"]["ci95"])


def test_bootstrap_point_matches_rank_block(df):
    out = h5_extras.h1_bootstrap(df, B=8, seed=1)
    res, _ = mt._rank_block(
        df[~df["scores_secondary"]],
        [a for a in mt.sort_cells(ARMS) if a[0] not in mt.RESTRICTED_AGE_FAMILIES])
    _, rho = res
    assert out["rho_rmse_crps"]["point"] == pytest.approx(rho["crps_logmx"], abs=1e-6)
    assert out["rho_rmse_logscore"]["point"] == pytest.approx(rho["poisson_log_score"], abs=1e-6)
    lo, hi = out["rho_rmse_crps"]["ci95"]
    assert -1.0 <= lo <= hi <= 1.0


JSON_PATH = ROOT / "results" / "h5_extras.json"


@pytest.mark.skipif(not JSON_PATH.exists(), reason="results/h5_extras.json not generated")
def test_committed_json_keys_and_ranges():
    j = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    assert j["script"] == "scripts/h5_extras.py"
    assert j["regime"] == "shift"
    arms = j["annuity_shortfall"]["arms"]
    assert arms, "no arms tabulated"
    for r in arms:
        assert {"model", "mechanism", "n", "n_admissible", "n_err",
                "upper_exceed_share", "upper_mean_rel_shortfall",
                "lower_exceed_share", "lower_mean_rel_shortfall"} <= set(r)
        for side in ("upper", "lower"):
            s = r[f"{side}_exceed_share"]
            assert s is None or 0.0 <= s <= 1.0
        assert r["n_admissible"] <= r["n"]
    boot = j["h1_rank_correlation"]
    assert boot["B"] == 2000 and boot["seed"] == 20260830
    for k in ("rho_rmse_crps", "rho_rmse_logscore"):
        lo, hi = boot[k]["ci95"]
        assert -1.0 <= lo <= hi <= 1.0
        assert -1.0 <= boot[k]["point"] <= 1.0
