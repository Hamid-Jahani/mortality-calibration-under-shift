"""Runner integration tests on the SYNTHETIC Poisson-LC truth only.

Real data may not flow through the runner until PREREGISTRATION.md validation
gate 2 (oracle parity) closes; these tests exercise the full sweep machinery
on worlds simulated from the known DGP of test_synthetic_calibration.py.
"""
import numpy as np
import pandas as pd
import pytest

from mortcal.runner import (
    ADMISSIBLE,
    CONFORMAL_MECHANISMS,
    MECHANISMS,
    MODELS,
    InadmissibleCellError,
    check_admissible,
    run_cell,
    run_regime,
)
from mortcal.splits import SHIFT, Regime

from test_synthetic_calibration import H, N_AGES, T_TRAIN, simulate_plc

EXPECTED_KEYS = {
    "rmse_logmx", "mae_logmx", "crps_logmx", "poisson_log_score",
    "coverage_50", "coverage_80", "coverage_95",
    "winkler_50", "winkler_80", "winkler_95",
    "joint_path_coverage_95", "pit_ks_stat",
    "e0_mean", "e0_q025", "e0_q975", "e0_obs",
    "e65_mean", "e65_q025", "e65_q975", "e65_obs",
    "ann65_mean", "ann65_q025", "ann65_q975", "ann65_obs",
    "scores_secondary",
}

FIRST_YEAR = 1940  # synthetic panel years FIRST_YEAR .. FIRST_YEAR+T_TRAIN+H-1


def _one_world(seed):
    """Train matrices + observed test window drawn from the known DGP."""
    rng = np.random.default_rng(seed)
    (D, E), true_mx = simulate_plc(rng)          # D,E [ages, T]; true_mx [H, ages]
    obs_E = np.full((H, N_AGES), 1e5)
    obs_D = rng.poisson(obs_E * true_mx).astype(float)
    return D, E, obs_D, obs_E


def _synthetic_panel(pops=("SYN_A", "SYN_B"), sex="f"):
    """Tidy 2-population panel (train + observed test years) from the DGP."""
    frames = []
    for i, pop in enumerate(pops):
        D, E, obs_D, obs_E = _one_world(1000 + i)
        D_all = np.concatenate([D, obs_D.T], axis=1)      # [ages, T+H]
        E_all = np.concatenate([E, obs_E.T], axis=1)
        ages, years = np.arange(N_AGES), FIRST_YEAR + np.arange(T_TRAIN + H)
        aa, yy = np.meshgrid(ages, years, indexing="ij")
        frames.append(pd.DataFrame({
            "pop": pop, "year": yy.ravel(), "age": aa.ravel(), "sex": sex,
            "D": D_all.ravel(), "E": E_all.ravel(),
        }))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# run_cell
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mechanism", ["native", "split_conf"])
def test_run_cell_plc_metrics_present_and_finite(mechanism):
    D, E, obs_D, obs_E = _one_world(42)
    h = 5
    out = run_cell(D, E, "PLC", mechanism, h, 400,
                   np.random.default_rng(7),
                   obs_D=obs_D[:h], obs_E=obs_E[:h])
    assert set(out) == EXPECTED_KEYS
    for k, v in out.items():
        if k == "scores_secondary":
            assert isinstance(v, bool)
            assert v is (mechanism in CONFORMAL_MECHANISMS)
        else:
            assert isinstance(v, float), f"{k} is {type(v)}"
            assert np.isfinite(v), f"{k} = {v}"
    # coverage/quantile sanity: correctly-specified DGP, in-DGP test window
    assert 0.0 <= out["coverage_95"] <= 1.0
    assert out["e0_q025"] <= out["e0_mean"] <= out["e0_q975"]
    assert out["ann65_q025"] <= out["ann65_q975"]


def test_run_cell_rejects_unknown_and_requires_obs():
    D, E, obs_D, obs_E = _one_world(3)
    with pytest.raises(InadmissibleCellError, match="unknown mechanism"):
        run_cell(D, E, "PLC", "dropout", 3, 50, np.random.default_rng(0),
                 obs_D=obs_D[:3], obs_E=obs_E[:3])
    with pytest.raises(InadmissibleCellError, match="unknown model"):
        run_cell(D, E, "GP", "native", 3, 50, np.random.default_rng(0),
                 obs_D=obs_D[:3], obs_E=obs_E[:3])
    with pytest.raises(ValueError, match="obs_D and obs_E"):
        run_cell(D, E, "PLC", "native", 3, 50, np.random.default_rng(0))


def test_pboot_without_fitted_mx_raises_clear_error():
    """Admissible per GRID.md but CBD lacks fitted_mx: the error must say so."""
    D, E, obs_D, obs_E = _one_world(4)
    with pytest.raises(NotImplementedError, match="fitted_mx"):
        run_cell(D, E, "CBD", "pboot", 3, 50, np.random.default_rng(0),
                 obs_D=obs_D[:3], obs_E=obs_E[:3])


def test_admissible_grid_matches_registries():
    for m in MODELS:
        for u in MECHANISMS:
            check_admissible(m, u)              # classical grid: all admissible
    assert ("LC", "native") in ADMISSIBLE
    with pytest.raises(InadmissibleCellError):
        check_admissible("LC", "deep_ensemble")


# ---------------------------------------------------------------------------
# run_regime
# ---------------------------------------------------------------------------

def test_run_regime_miniature_synthetic_sweep(tmp_path):
    panel = _synthetic_panel()
    pops = ("SYN_A", "SYN_B")
    last_train = FIRST_YEAR + T_TRAIN - 1                 # 1999
    regimes = tuple(
        Regime(name="synthetic", train_max_year=o,
               test_years=tuple(range(o + 1, o + 4)),     # h = 3
               horizons=(1, 2, 3), pops=pops)
        for o in (last_train, last_train + 2)             # 2 origins
    )
    out_path = tmp_path / "mini_sweep.parquet"
    models, mechs = ["LC", "PLC"], ["native", "split_conf"]
    df = run_regime(panel, regimes, models, mechs,
                    n_samples=300, out_path=out_path)

    expected = len(regimes) * len(pops) * 1 * len(models) * len(mechs)  # 16
    assert len(df) == expected
    assert df["error"].isna().all(), df.loc[df["error"].notna(), "error"].tolist()

    back = pd.read_parquet(out_path)
    assert len(back) == expected
    key_cols = ["regime", "pop", "sex", "origin", "model", "mechanism"]
    assert not back.duplicated(subset=key_cols).any()
    for k in EXPECTED_KEYS - {"scores_secondary"}:
        assert np.isfinite(back[k]).all(), k
    assert back["scores_secondary"].astype(bool).eq(
        back["mechanism"].isin(list(CONFORMAL_MECHANISMS))).all()
    assert set(back["origin"]) == {last_train, last_train + 2}


def test_run_regime_refuses_real_regimes_while_gate_open(tmp_path):
    """PREREGISTRATION.md gate 2 (oracle parity) is open: real regimes blocked."""
    panel = _synthetic_panel()
    with pytest.raises(RuntimeError, match="gate 2"):
        run_regime(panel, SHIFT, ["PLC"], ["native"],
                   n_samples=10, out_path=tmp_path / "x.parquet")


def test_run_regime_records_fit_errors_instead_of_crashing(tmp_path):
    """A cell whose fit raises becomes an error row, not a dead sweep."""
    panel = _synthetic_panel(pops=("SYN_A",))
    # Truncate to 15 training years: SplitConformalMx needs T - 8 >= 10 and
    # must raise; native PLC on 15 years still fits fine.
    origin = FIRST_YEAR + 14
    panel = panel[panel["year"] <= origin + 3]
    reg = Regime(name="synthetic", train_max_year=origin,
                 test_years=tuple(range(origin + 1, origin + 4)),
                 horizons=(1, 2, 3), pops=("SYN_A",))
    msgs = []
    df = run_regime(panel, reg, ["PLC"], ["native", "split_conf"],
                    n_samples=100, out_path=tmp_path / "err.parquet",
                    log=msgs.append)
    assert len(df) == 2
    ok = df[df["mechanism"] == "native"].iloc[0]
    bad = df[df["mechanism"] == "split_conf"].iloc[0]
    assert pd.isna(ok["error"]) and np.isfinite(ok["crps_logmx"])
    assert isinstance(bad["error"], str) and "ValueError" in bad["error"]
    assert len(msgs) == 1 and "split_conf" in msgs[0]
