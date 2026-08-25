"""CBD (M5) validation gate: the full chain — per-year OLS calibration,
bivariate RWD on (k1, k2), pathwise sampling — must recover the parameters of,
and attain nominal coverage under, a KNOWN CBD data-generating process with
Poisson observation noise: D ~ Poisson(E * m), m = -log(1 - q) — the same
constant-force q<->m mapping the estimator inverts, so the model is correctly
specified and nominal coverage is the pass criterion (methodology rule 5)."""
import numpy as np
import pytest

from mortcal.eval import interval_coverage
from mortcal.models.cbd import CBD

AGES = np.arange(55, 100)                     # CBD's home range: 45 old ages
XBAR = AGES.mean()                            # 77.0
T_TRAIN, H, N_SAMPLES, N_WORLDS = 40, 10, 1000, 25

K0 = np.array([-3.1, 0.10])                   # logit q(77) ~ -3.1; +0.10 per year of age
MU = np.array([-0.020, 0.0003])               # improvement drift on (k1, k2)
_SD = np.array([0.05, 0.001])
_RHO = -0.3
SIGMA = np.array([[_SD[0] ** 2, _RHO * _SD[0] * _SD[1]],
                  [_RHO * _SD[0] * _SD[1], _SD[1] ** 2]])


def simulate_cbd(rng, t_total=T_TRAIN + H, expo=1e5, ages=AGES, xbar=XBAR):
    """One world from the known CBD DGP.

    Returns (D, E, m, K): deaths/exposures [ages, t_total], true central rates
    m [ages, t_total], true factor path K [t_total, 2]. The logit surface is
    centred at `xbar` regardless of which ages are materialised, so a full-age
    matrix stays consistent with its 55-99 sub-block.
    """
    inc = rng.multivariate_normal(MU, SIGMA, size=t_total)
    K = K0[None, :] + np.cumsum(inc, axis=0)                    # [T, 2]
    z = K[:, 0][None, :] + np.outer(ages - xbar, K[:, 1])       # logit q, [ages, T]
    m = np.logaddexp(0.0, z)                                    # -log(1-q)
    E = np.full((len(ages), t_total), float(expo))
    D = rng.poisson(E * m).astype(float)
    return D, E, m, K


def test_parameter_recovery():
    """(k1, k2) recovered from Poisson-noised data: correlation with the true
    factor path > 0.99, and near-unbiased in level (same xbar by construction)."""
    rng = np.random.default_rng(1)
    D, E, _, K = simulate_cbd(rng, t_total=60, expo=1e6)
    model = CBD().fit(D, E, age0=55)
    r1 = float(np.corrcoef(model.k1, K[:, 0])[0, 1])
    r2 = float(np.corrcoef(model.k2, K[:, 1])[0, 1])
    assert r1 > 0.99, f"corr(k1_hat, k1_true) = {r1:.4f}"
    assert r2 > 0.99, f"corr(k2_hat, k2_true) = {r2:.4f}"
    assert np.max(np.abs(model.k1 - K[:, 0])) < 0.02
    assert np.max(np.abs(model.k2 - K[:, 1])) < 0.001


def test_nominal_coverage_on_known_cbd_dgp():
    """Correctly specified model on its own DGP: mean marginal 95% coverage of
    future m_x over 25 worlds must land in [0.90, 0.99] (Monte-Carlo band:
    worlds are internally correlated across ages and horizons)."""
    rng = np.random.default_rng(20260825)
    covs = []
    for _ in range(N_WORLDS):
        D, E, m, _ = simulate_cbd(rng)
        model = CBD().fit(D[:, :T_TRAIN], E[:, :T_TRAIN], age0=55)
        samples = np.log(model.sample_mx(H, N_SAMPLES, rng))    # [n, H, ages]
        cov, _ = interval_coverage(samples, np.log(m[:, T_TRAIN:].T), 0.95)
        covs.append(float(cov.mean()))
    c = float(np.mean(covs))
    assert 0.90 <= c <= 0.99, f"nominal 95% attained {c:.3f}"


def test_ages_kwarg_selects_rows():
    """HMD-style full matrix (rows = ages 0-99): fitting ages 55-99 via the
    ages/age0 kwargs must equal fitting the pre-sliced sub-matrix, and
    sample_mx must return [n, h, len(ages)]."""
    rng = np.random.default_rng(2)
    D, E, _, _ = simulate_cbd(rng, ages=np.arange(100), xbar=XBAR)
    a = CBD().fit(D, E, ages=AGES, age0=0)
    b = CBD().fit(D[55:], E[55:], age0=55)
    np.testing.assert_allclose(a.k1, b.k1)
    np.testing.assert_allclose(a.k2, b.k2)
    s = a.sample_mx(3, 7, np.random.default_rng(0))
    assert s.shape == (7, 3, len(AGES))
    assert np.all(np.isfinite(s)) and np.all(s > 0)


def test_interval_width_grows_with_horizon():
    """Pathwise accumulation: predictive spread must widen with horizon
    (variance h*Sigma + h^2 * drift term); flat width would mean horizons are
    sampled independently and joint path coverage would be meaningless."""
    rng = np.random.default_rng(3)
    D, E, _, _ = simulate_cbd(rng)
    model = CBD().fit(D[:, :T_TRAIN], E[:, :T_TRAIN], age0=55)
    s = np.log(model.sample_mx(H, 4000, rng))
    width = np.quantile(s, 0.975, axis=0) - np.quantile(s, 0.025, axis=0)
    w = width.mean(axis=1)                                      # [H]
    assert np.all(np.diff(w) > 0), f"widths not increasing: {np.round(w, 3)}"


def test_input_validation():
    rng = np.random.default_rng(4)
    D, E, _, _ = simulate_cbd(rng)
    with pytest.raises(ValueError):                             # too few years
        CBD().fit(D[:, :2], E[:, :2], age0=55)
    with pytest.raises(ValueError):                             # ages outside rows
        CBD().fit(D, E, ages=np.arange(50, 100), age0=55)
