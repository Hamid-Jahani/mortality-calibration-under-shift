"""Runner integration tests on SYNTHETIC Poisson-LC truth only.

Real data may not flow through the runner without an explicit
``allow_real=True``; these tests exercise the full sweep machinery on worlds
simulated from the DGP of test_synthetic_calibration.py, extended to the
HMD age range 0-99 so every age band, the CBD age restriction (55+) and the
e65 / ä65 columns are exercised exactly as on real panels.
"""
import json

import numpy as np
import pandas as pd
import pytest

from mortcal.lifetable import annuity_factor, life_expectancy
from mortcal.runner import (
    ADMISSIBLE,
    AGE_BANDS,
    CONFORMAL_MECHANISMS,
    MECHANISMS,
    MODEL_KWARGS,
    MODELS,
    InadmissibleCellError,
    build_estimator,
    check_admissible,
    run_cell,
    run_regime,
)
from mortcal.splits import SHIFT, Regime
from mortcal.uq import PoissonBootstrap

from test_synthetic_calibration import H, T_TRAIN

N_AGES = 100          # full HMD single-year age range 0-99
FIRST_YEAR = 1940     # synthetic panel years FIRST_YEAR .. FIRST_YEAR+T_TRAIN+H-1

BAND_TAGS = [f"band{lo}_{hi}" for lo, hi in AGE_BANDS]
SCALAR_KEYS = {
    "n_ages_scored", "n_cells", "n_zero_death_cells",
    "rmse_logmx", "mae_logmx", "crps_logmx", "crps_counts", "poisson_log_score",
    "coverage_50", "coverage_80", "coverage_95",
    "winkler_50", "winkler_80", "winkler_95",
    "joint_path_coverage_95", "pit_ks_stat", "pit_ks_pvalue",
    "derived_age_lo", "derived_age_hi",
    "murphy_reliability", "murphy_resolution", "murphy_uncertainty", "murphy_brier",
    "murphy_pit_reliability", "murphy_pit_resolution", "murphy_pit_uncertainty",
}
BAND_KEYS = ({f"coverage_{lvl}_{b}" for lvl in (50, 80, 95) for b in BAND_TAGS}
             | {f"pit_ks_{b}" for b in BAND_TAGS})
INT_KEYS = ("n_ages_scored", "n_cells", "n_zero_death_cells",
            "derived_age_lo", "derived_age_hi")

#: PREREGISTRATION-ADDENDUM-2 §3 — a conformal mechanism constructs ONE
#: interval, at 95%. Its 50% / 80% columns are registered not-applicable, so
#: they are NaN by design rather than missing or zero.
CONFORMAL_NA_KEYS = ({f"coverage_{lvl}" for lvl in (50, 80)}
                     | {f"winkler_{lvl}" for lvl in (50, 80)}
                     | {f"coverage_{lvl}_{b}" for lvl in (50, 80) for b in BAND_TAGS})
DERIVED_KEYS = {f"{q}_{s}" for q in ("e0", "e65", "ann65")
                for s in ("point", "q025", "q975", "obs", "error")}
JSON_KEYS = {"cov95_by_age", "pit_hist"}


def horizon_keys(h):
    return {f"{p}_h{k}" for p in ("crps", "logscore", "coverage95", "winkler95")
            for k in range(1, h + 1)}


def expected_keys(h):
    return (SCALAR_KEYS | BAND_KEYS | DERIVED_KEYS | JSON_KEYS
            | horizon_keys(h) | {"scores_secondary"})


def simulate_world(rng, n_ages=N_AGES):
    """The Poisson-LC DGP of test_synthetic_calibration.simulate_plc on a
    0..n_ages-1 age range: alpha rises to ~ -2.0 (m_99 ~ 0.13), beta peaks
    at young ages. Returns train (D, E) and the true future m_x [H, ages]."""
    ages = np.arange(n_ages)
    alpha = -7.5 + 5.5 * (ages / n_ages) ** 1.3
    beta = np.exp(-0.5 * ((ages - 12) / 14.0) ** 2)
    beta = beta / beta.sum()
    k = np.cumsum(-1.2 + rng.normal(0, 0.9, T_TRAIN + H))
    k = k - k[:T_TRAIN].mean()
    mx_all = np.exp(alpha[:, None] + np.outer(beta, k))
    E = np.full((n_ages, T_TRAIN + H), 1e5)
    D = rng.poisson(E * mx_all).astype(float)
    return (D[:, :T_TRAIN], E[:, :T_TRAIN]), mx_all[:, T_TRAIN:].T


def _one_world(seed):
    """Train matrices + observed test window drawn from the known DGP."""
    rng = np.random.default_rng(seed)
    (D, E), true_mx = simulate_world(rng)        # D,E [ages, T]; true_mx [H, ages]
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


def _assert_types(out, mechanism, nan_ok=()):
    """Every value has the documented python type; floats finite unless listed.

    Conformal mechanisms carry the addendum-2 §3 not-applicable columns, which
    are NaN by design; they are added to ``nan_ok`` automatically so no caller
    has to remember.
    """
    nan_ok = set(nan_ok)
    if mechanism in CONFORMAL_MECHANISMS:
        nan_ok |= CONFORMAL_NA_KEYS
    for k, v in out.items():
        if k == "scores_secondary":
            assert isinstance(v, bool)
            assert v is (mechanism in CONFORMAL_MECHANISMS)
        elif k in JSON_KEYS:
            assert isinstance(v, str)
            json.loads(v)
        elif k in INT_KEYS:
            assert isinstance(v, int)
        else:
            assert isinstance(v, float), f"{k} is {type(v)}"
            if k in nan_ok:
                assert np.isnan(v), f"{k} = {v} should be NaN by design"
            else:
                assert np.isfinite(v), f"{k} = {v}"


def _assert_internal_consistency(out, h):
    """Means equal the average of their per-horizon series; Murphy identities."""
    for mean_key, prefix in (("crps_logmx", "crps"), ("poisson_log_score", "logscore"),
                             ("coverage_95", "coverage95"), ("winkler_95", "winkler95")):
        series = [out[f"{prefix}_h{k}"] for k in range(1, h + 1)]
        assert np.isclose(out[mean_key], np.mean(series)), mean_key
    # Murphy on the 95% hit indicators with the constant forecast 0.95:
    # reliability is the squared coverage gap, resolution is zero, and the
    # three-term identity is exact.
    c = out["coverage_95"]
    assert np.isclose(out["murphy_reliability"], (0.95 - c) ** 2)
    assert np.isclose(out["murphy_resolution"], 0.0)
    assert np.isclose(out["murphy_uncertainty"], c * (1 - c))
    assert np.isclose(out["murphy_brier"], out["murphy_reliability"]
                      - out["murphy_resolution"] + out["murphy_uncertainty"])
    # PIT-scale decomposition: brier = 1 - 1/K = uncertainty, identity exact.
    assert np.isclose(out["murphy_pit_uncertainty"], 0.9)
    assert np.isclose(out["murphy_pit_reliability"], out["murphy_pit_resolution"])
    hist = json.loads(out["pit_hist"])
    assert len(hist) == 10 and np.isclose(sum(hist), 1.0)
    assert np.isclose(out["murphy_pit_reliability"],
                      sum((g - 0.1) ** 2 for g in hist))
    # derived point functional (median) sits inside its own interval
    for q in ("e0", "e65", "ann65"):
        if np.isfinite(out[f"{q}_point"]):
            assert out[f"{q}_q025"] <= out[f"{q}_point"] <= out[f"{q}_q975"]
            assert np.isclose(out[f"{q}_error"], out[f"{q}_point"] - out[f"{q}_obs"])


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
    assert set(out) == expected_keys(h)
    _assert_types(out, mechanism)
    _assert_internal_consistency(out, h)
    assert out["n_ages_scored"] == N_AGES and out["n_cells"] == h * N_AGES
    # coverage/quantile sanity: correctly-specified DGP, in-DGP test window
    assert 0.0 <= out["coverage_95"] <= 1.0
    assert out["e0_q025"] <= out["e0_point"] <= out["e0_q975"]
    assert out["ann65_q025"] <= out["ann65_q975"]
    # by-age curve: one finite entry per panel age, each a mean of h indicators
    curve = json.loads(out["cov95_by_age"])
    assert len(curve) == N_AGES and all(v is not None for v in curve)
    assert np.isclose(np.mean(curve), out["coverage_95"])
    assert all(np.isclose(v * h, round(v * h)) for v in curve)
    # band coverages average (age-weighted) to the overall coverage
    n_band = [25, 40, 35]
    assert np.isclose(
        sum(out[f"coverage_95_{b}"] * n for b, n in zip(BAND_TAGS, n_band)) / N_AGES,
        out["coverage_95"])


def test_run_cell_cbd_native_masks_ages_below_55():
    """CBD(age_min=55) via MODEL_KWARGS: metrics over ages 55-99 only, with
    the denominators recorded; e0 undefined (NaN) but e65 / ä65 exact from
    the truncated table; band 0-24 has nothing to score."""
    D, E, obs_D, obs_E = _one_world(11)
    h = 5
    out = run_cell(D, E, "CBD", "native", h, 400, np.random.default_rng(3),
                   obs_D=obs_D[:h], obs_E=obs_E[:h])
    assert set(out) == expected_keys(h)
    nan_by_design = {"e0_point", "e0_q025", "e0_q975", "e0_error",
                     "coverage_50_band0_24", "coverage_80_band0_24",
                     "coverage_95_band0_24", "pit_ks_band0_24"}
    _assert_types(out, "native", nan_ok=nan_by_design)
    _assert_internal_consistency(out, h)
    assert out["n_ages_scored"] == 45 and out["n_cells"] == h * 45
    assert np.isfinite(out["e0_obs"])           # observed panel is always full
    curve = json.loads(out["cov95_by_age"])
    assert len(curve) == N_AGES
    assert all(v is None for v in curve[:55]) and all(v is not None for v in curve[55:])
    assert np.isclose(np.mean(curve[55:]), out["coverage_95"])
    # band 25-64 = ages 55-64 only (10 ages), band 65-99 = 35 ages
    assert np.isclose((10 * out["coverage_95_band25_64"]
                       + 35 * out["coverage_95_band65_99"]) / 45, out["coverage_95"])
    assert 0.0 <= out["coverage_95"] <= 1.0


def test_truncated_table_invariance_used_for_masked_e65_and_annuity():
    """The masked-age treatment in _derived_quantities rests on e_x and ä_x
    being ratios to l_x: a table starting at 55 must give the SAME e65 / ä65
    as the full 0-99 table, infant rule on its first row notwithstanding."""
    _D, _E, obs_D, obs_E = _one_world(5)
    mx = obs_D[0] / obs_E[0]
    assert np.isclose(life_expectancy(mx, 65), life_expectancy(mx[55:], 10), rtol=1e-12)
    assert np.isclose(annuity_factor(mx, x0=65, i=0.02),
                      annuity_factor(mx[55:], x0=10, i=0.02), rtol=1e-12)
    mx2 = np.stack([mx, mx * 1.1])                    # vectorised path too
    np.testing.assert_allclose(life_expectancy(mx2, 65),
                               life_expectancy(mx2[:, 55:], 10), rtol=1e-12)


def test_run_cell_rh_pboot_tiny_B():
    """(RH, pboot) — a family without sample_deaths on a wrapper: death
    counts are composed as Poisson(E * sample_mx); every column finite."""
    D, E, obs_D, obs_E = _one_world(8)
    h = 4
    out = run_cell(D, E, "RH", "pboot", h, 120, np.random.default_rng(1),
                   obs_D=obs_D[:h], obs_E=obs_E[:h],
                   mech_kwargs={"B": 3, "n_inner": 2})
    assert set(out) == expected_keys(h)
    _assert_types(out, "pboot")
    _assert_internal_consistency(out, h)
    assert out["n_cells"] == h * N_AGES
    assert out["crps_counts"] > 0.0


@pytest.mark.parametrize("h", [3, 7])
def test_per_horizon_column_count_equals_H(h):
    D, E, obs_D, obs_E = _one_world(21)
    out = run_cell(D, E, "PLC", "native", h, 150, np.random.default_rng(0),
                   obs_D=obs_D[:h], obs_E=obs_E[:h])
    for prefix in ("crps", "logscore", "coverage95", "winkler95"):
        cols = {k for k in out if k.startswith(f"{prefix}_h")}
        assert cols == {f"{prefix}_h{k}" for k in range(1, h + 1)}
        assert len(cols) == h


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


def test_pboot_constructible_for_every_classical_family():
    """All five families implement fitted_mx(): pboot is admissible AND
    constructible for each, with the family kwargs forwarded to every refit."""
    for m in MODELS:
        est = build_estimator(m, "pboot", {"B": 2})
        assert isinstance(est, PoissonBootstrap) and est.B == 2
        assert est.model_kwargs == MODEL_KWARGS.get(m, {})
    assert MODEL_KWARGS["CBD"] == {"age_min": 55}
    assert build_estimator("CBD", "native").age_min == 55
    split = build_estimator("CBD", "split_conf")
    assert split.base_factory().age_min == 55       # partial carries the kwarg


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
    numeric = expected_keys(3) - {"scores_secondary"} - JSON_KEYS
    is_conf = back["mechanism"].isin(list(CONFORMAL_MECHANISMS))
    for k in numeric:
        col = back[k].astype(float)
        if k in CONFORMAL_NA_KEYS:
            # addendum 2 §3: N/A on conformal rows, finite everywhere else
            assert np.isfinite(col[~is_conf]).all(), k
            assert col[is_conf].isna().all(), k
        else:
            assert np.isfinite(col).all(), k
    assert set(back.columns) >= expected_keys(3)
    assert (back["n_cells"] == 3 * N_AGES).all()
    assert back["scores_secondary"].astype(bool).eq(
        back["mechanism"].isin(list(CONFORMAL_MECHANISMS))).all()
    assert set(back["origin"]) == {last_train, last_train + 2}
    # JSON vectors survive the parquet round trip
    assert all(len(json.loads(s)) == N_AGES for s in back["cov95_by_age"])
    assert all(len(json.loads(s)) == 10 for s in back["pit_hist"])


def test_run_regime_refuses_real_regimes_without_allow_real(tmp_path):
    """Real regimes require the explicit, auditable allow_real=True."""
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
    assert pd.isna(bad["n_cells"])
    assert len(msgs) == 1 and "split_conf" in msgs[0]
