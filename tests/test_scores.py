"""Validation of scoring rules against closed forms — rule 5 at unit level:
if these fail, no downstream coverage number is trustworthy."""
import numpy as np
import pytest
from scipy import stats

from mortcal.eval import (
    crps_sample, log_score_poisson, interval_coverage,
    winkler_score, pit_values, joint_path_coverage,
    round_deaths, crps_counts, murphy_decomposition, murphy_pit,
)

RNG = np.random.default_rng(42)


def test_crps_matches_gaussian_closed_form():
    # CRPS(N(mu,s), y) = s * [ z(2Phi(z)-1) + 2phi(z) - 1/sqrt(pi) ]
    mu, sig, y = 1.3, 0.7, 2.1
    z = (y - mu) / sig
    closed = sig * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    samples = RNG.normal(mu, sig, size=(40000, 1))
    est = crps_sample(samples, np.array([y]))[0]
    assert est == pytest.approx(closed, rel=0.02)


def test_crps_zero_for_point_mass_on_truth():
    samples = np.full((100, 3), 5.0)
    assert np.allclose(crps_sample(samples, np.full(3, 5.0)), 0.0)


def test_log_score_poisson_single_and_mixture_agree():
    lam = np.array([50.0, 200.0])
    d = np.array([48.0, 210.0])
    single = log_score_poisson(lam, d)
    mixture = log_score_poisson(np.tile(lam, (7, 1)), d)  # 7 identical components
    direct = -stats.poisson.logpmf(d, lam)
    assert np.allclose(single, direct)
    assert np.allclose(mixture, direct, atol=1e-10)


def test_interval_coverage_hits_nominal_under_truth():
    samples = RNG.normal(0, 1, size=(4000, 5000))
    truth = RNG.normal(0, 1, size=5000)
    covered, width = interval_coverage(samples, truth, 0.95)
    assert covered.mean() == pytest.approx(0.95, abs=0.01)
    assert width.mean() == pytest.approx(2 * 1.96, rel=0.02)


def test_winkler_penalises_misses_not_hits():
    samples = RNG.normal(0, 1, size=(20000, 1))
    inside = winkler_score(samples, np.array([0.0]), 0.95)[0]
    outside = winkler_score(samples, np.array([4.0]), 0.95)[0]
    assert inside == pytest.approx(2 * 1.96, rel=0.03)  # width only
    assert outside > inside + 2 / 0.05 * 1.5  # miss penalty dominates


def test_pit_uniform_under_correct_model():
    samples = RNG.normal(0, 1, size=(999, 3000))
    truth = RNG.normal(0, 1, size=3000)
    pit = pit_values(samples, truth, rng=np.random.default_rng(7))
    ks = stats.kstest(pit, "uniform")
    assert ks.pvalue > 0.01


def test_joint_path_coverage_below_marginal_analytic():
    # H independent horizons: joint = level^H
    H, units, level = 5, 4000, 0.9
    samples = RNG.normal(0, 1, size=(2000, H, units))
    truth = RNG.normal(0, 1, size=(H, units))
    joint = joint_path_coverage(samples, truth, level)
    assert joint == pytest.approx(level ** H, abs=0.03)
    covered, _ = interval_coverage(samples, truth, level)
    assert covered.mean() > joint + 0.2  # the H3 gap, visible even in toy form


# --- rounding convention ----------------------------------------------------

def test_round_deaths_is_half_up_not_bankers():
    d = np.array([2.5, 3.5, 2.4, 2.6, 0.5, 47.0])
    assert np.array_equal(round_deaths(d), [3.0, 4.0, 2.0, 3.0, 1.0, 47.0])
    assert round_deaths(2.5) == 3.0 and round_deaths(3.5) == 4.0
    # the convention that was rejected: numpy rounds halves to even
    assert np.round(2.5) == 2.0 and np.round(3.5) == 4.0


def test_log_score_poisson_applies_half_up_rounding():
    lam = np.array([10.0, 10.0])
    d = np.array([2.5, 3.5])  # Lexis-split halves
    want = -stats.poisson.logpmf(np.array([3, 4]), lam)
    assert np.allclose(log_score_poisson(lam, d), want)
    assert np.allclose(log_score_poisson(np.tile(lam, (5, 1)), d), want, atol=1e-10)
    assert not np.allclose(log_score_poisson(lam, d)[0], -stats.poisson.logpmf(2, 10.0))


# --- CRPS on counts ----------------------------------------------------------

def test_crps_counts_matches_crps_sample_and_does_not_round():
    rng = np.random.default_rng(3)
    lam = rng.uniform(20, 300, size=(4, 6))
    samples = rng.poisson(lam, size=(500, 4, 6)).astype(float)
    obs = np.floor(lam + rng.normal(0, 3, size=(4, 6))) + 0.5  # fractional truth
    assert np.array_equal(crps_counts(samples, obs), crps_sample(samples, obs))
    assert np.array_equal(crps_counts(samples.tolist(), obs.tolist()),
                          crps_sample(samples, obs))
    # the sensitivity companion is rounding-free by design
    assert not np.allclose(crps_counts(samples, obs), crps_sample(samples, round_deaths(obs)))


# --- Murphy decomposition ----------------------------------------------------

def test_murphy_three_terms_sum_to_brier_for_discrete_forecasts():
    rng = np.random.default_rng(11)
    p = rng.choice([0.5, 0.8, 0.95], size=5000)
    o = rng.uniform(size=5000) < p - 0.1        # over-confident by 0.1 everywhere
    for nb in (10, None):
        m = murphy_decomposition(p, o, n_bins=nb)
        assert m["reliability"] - m["resolution"] + m["uncertainty"] == pytest.approx(
            m["brier"], abs=1e-12)
        assert m["within_bin_variance"] == pytest.approx(0.0, abs=1e-20)
        assert m["within_bin_covariance"] == pytest.approx(0.0, abs=1e-20)
        assert m["brier"] == pytest.approx(np.mean((p - o) ** 2))
        assert m["n"] == 5000
    assert m["reliability"] == pytest.approx(0.01, abs=0.004)   # (0.1)^2 gap
    assert m["resolution"] > 0.02                              # hit rate moves with level


def test_murphy_five_term_identity_for_continuous_forecasts():
    rng = np.random.default_rng(12)
    p = rng.uniform(size=5000)
    o = rng.uniform(size=5000) < p                # calibrated
    m = murphy_decomposition(p, o, n_bins=10)
    five = (m["reliability"] - m["resolution"] + m["uncertainty"]
            + m["within_bin_variance"] - m["within_bin_covariance"])
    assert five == pytest.approx(m["brier"], abs=1e-12)
    assert m["within_bin_variance"] > 0.0
    assert m["reliability"] < 0.005
    assert m["resolution"] == pytest.approx(1 / 12, abs=0.02)  # Var(p) up to binning


def test_murphy_constant_nominal_reduces_to_coverage_gap():
    rng = np.random.default_rng(13)
    o = rng.uniform(size=4000) < 0.85
    m = murphy_decomposition(np.full(4000, 0.95), o)
    cov = o.mean()
    assert m["reliability"] == pytest.approx((0.95 - cov) ** 2, abs=1e-12)
    assert m["resolution"] == pytest.approx(0.0, abs=1e-15)
    assert m["uncertainty"] == pytest.approx(cov * (1 - cov))


def test_murphy_rejects_bad_inputs():
    with pytest.raises(ValueError):
        murphy_decomposition([0.5, 1.2], [1, 0])
    with pytest.raises(ValueError):
        murphy_decomposition([0.5, 0.5], [1, 2])
    with pytest.raises(ValueError):
        murphy_decomposition([0.5, 0.5, 0.5], [1, 0])
    m = murphy_decomposition([0.5, np.nan, 0.5], [1, 0, 0])
    assert m["n"] == 2


def test_murphy_pit_flat_is_reliable_hump_is_not():
    rng = np.random.default_rng(14)
    flat = murphy_pit(rng.uniform(size=20000))
    hump = murphy_pit(rng.beta(3, 3, size=20000))
    assert flat["reliability"] < 5e-4   # E = (1 - 1/K)/N = 4.5e-5
    assert hump["reliability"] > 0.01   # analytic ~0.041 for Beta(3,3), K=10
    for m in (flat, hump):
        assert m["reliability"] - m["resolution"] + m["uncertainty"] == pytest.approx(
            m["brier"], abs=1e-12)
        assert m["uncertainty"] == pytest.approx(0.9)
        assert m["hist"].shape == (10,) and m["hist"].sum() == pytest.approx(1.0)
        assert m["n"] == 20000
    assert np.argmax(hump["hist"]) in (4, 5)
    # boundary value 1.0 belongs to the last bin, not an 11th one
    assert murphy_pit(np.array([0.0, 1.0, 0.999]), n_bins=10)["hist"][-1] == pytest.approx(2 / 3)
