"""Validation of scoring rules against closed forms — rule 5 at unit level:
if these fail, no downstream coverage number is trustworthy."""
import numpy as np
import pytest
from scipy import stats

from mortcal.eval import (
    crps_sample, log_score_poisson, interval_coverage,
    winkler_score, pit_values, joint_path_coverage,
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
