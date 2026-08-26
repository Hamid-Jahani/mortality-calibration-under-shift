"""Guards for PREREGISTRATION-ADDENDUM-2 — scoring-target clarifications.

Addendum 2 registers four rules. Nothing tested them, which is how the
registered protocol and the implementation drifted apart:

  1. Rate-scale scores evaluate the predictive law of the OBSERVED rate.
     Samples are Poisson-inclusive: D* ~ Poisson(E * m_x*), scored as log
     crude rates. Scoring latent m_x* against observed D/E counts Poisson
     observation noise as miscalibration.
  2. Zero-death cells use the half-count continuity correction max(D, 0.5)/E
     on BOTH sides, and are counted rather than dropped.
  3. Conformal cells are scored on their interval bounds at the construction
     level (95%); the 50% and 80% columns are not-applicable.
  4. PIT uniformity reports the KS statistic AND a (descriptive) p-value.

The first test is the one that matters: under addendum 2 a correctly
specified model must attain nominal coverage against the data it actually
observes, not merely against a latent truth available only in simulation.
That is validation gate 1 enforced through the real-data code path.
"""
import numpy as np
import pytest

from mortcal.runner import run_cell

from test_runner import N_AGES, _one_world
from test_synthetic_calibration import H

NOMINAL = 0.95
SEEDS = (11, 12, 13)


def _cell(model, mechanism, D, E, obs_D, obs_E, seed, n_samples=200):
    """run_cell with this repo's argument order, one place to keep it right."""
    return run_cell(D, E, model, mechanism, h=H, n_samples=n_samples,
                    rng=np.random.default_rng(seed), obs_D=obs_D, obs_E=obs_E)


# ---------------------------------------------------------------------------
# 1 - rate-scale scores evaluate the OBSERVED rate's predictive law
# ---------------------------------------------------------------------------

def test_correctly_specified_model_covers_nominally_against_observed_rates():
    """PLC fit to its own DGP must attain ~95% coverage of the OBSERVED rate.

    The samples carry parameter/process uncertainty; the observation carries
    Poisson noise. Only a Poisson-inclusive predictive law contains both, so
    this fails whenever the runner scores latent m_x* against observed D/E.
    """
    covs = []
    for seed in SEEDS:
        D, E, obs_D, obs_E = _one_world(seed)
        out = _cell("PLC", "native", D, E, obs_D, obs_E, seed, n_samples=400)
        covs.append(out["coverage_95"])
    mean_cov = float(np.mean(covs))
    assert 0.90 <= mean_cov <= 0.985, (
        f"coverage_95 = {mean_cov:.3f} over seeds {SEEDS}; a correctly "
        "specified model must be near nominal against the observable")


@pytest.mark.parametrize("level,tol", [(50, 0.08), (80, 0.07), (95, 0.05)])
def test_correctly_specified_model_covers_at_every_registered_level(level, tol):
    """Nominal coverage must hold at all three registered levels, not just 95%.

    Scoring latent m_x* against observed D/E under-covers at every level,
    increasingly so as the level widens.
    """
    covs = []
    for seed in SEEDS:
        D, E, obs_D, obs_E = _one_world(seed)
        out = _cell("PLC", "native", D, E, obs_D, obs_E, seed, n_samples=400)
        covs.append(out[f"coverage_{level}"])
    got = float(np.mean(covs))
    assert abs(got - level / 100.0) <= tol, (
        f"coverage_{level} = {got:.3f}, nominal {level / 100.0:.2f}")


# ---------------------------------------------------------------------------
# 2 - zero-death cells: half-count continuity correction, no masking
# ---------------------------------------------------------------------------

def test_log_crude_rate_applies_half_count_correction():
    from mortcal.eval.scores import log_crude_rate

    E = np.array([2100.0, 6000.0, 2100.0, 2100.0])
    D = np.array([0.0, 0.0, 1.0, 3.0])
    got = log_crude_rate(D, E)
    want = np.log(np.array([0.5, 0.5, 1.0, 3.0]) / E)
    np.testing.assert_allclose(got, want)


def test_log_crude_rate_never_returns_the_rate_floor():
    """log(1e-10) = -23.03 is the artefact addendum 2 exists to remove."""
    from mortcal.eval.scores import log_crude_rate

    got = log_crude_rate(np.zeros(3), np.array([1e3, 1e4, 1e5]))
    assert np.all(got > -15.0), got


def test_zero_death_cells_are_counted_not_dropped():
    """A zero-death cell stays scorable and is reported."""
    D, E, obs_D, obs_E = _one_world(31)
    obs_D = obs_D.copy()
    obs_D[0, 5] = 0.0
    obs_D[2, 7] = 0.0
    out = _cell("PLC", "native", D, E, obs_D, obs_E, 31, n_samples=200)
    assert out["n_zero_death_cells"] == 2
    assert out["n_ages_scored"] == N_AGES, "zero-death ages must not be masked"


def test_zero_death_cell_does_not_explode_the_rate_scale():
    """Injecting one zero-death cell must not move RMSE by orders of magnitude."""
    D, E, obs_D, obs_E = _one_world(41)
    base = _cell("PLC", "native", D, E, obs_D, obs_E, 41, n_samples=200)
    holed = obs_D.copy()
    holed[0, 9] = 0.0
    hit = _cell("PLC", "native", D, E, holed, obs_E, 41, n_samples=200)
    # measured 2026-08-26 with the 1e-10 rate floor: 0.124 -> 0.475, a 3.8x
    # blow-up from a single cell in 500. The half-count correction keeps it
    # within noise of the base value.
    assert hit["rmse_logmx"] < 1.5 * base["rmse_logmx"], (
        base["rmse_logmx"], hit["rmse_logmx"])


# ---------------------------------------------------------------------------
# 3 - conformal cells scored at their construction level only
# ---------------------------------------------------------------------------

def test_conformal_reports_only_the_construction_level_interval():
    D, E, obs_D, obs_E = _one_world(51)
    out = _cell("PLC", "split_conf", D, E, obs_D, obs_E, 51, n_samples=200)
    assert np.isfinite(out["coverage_95"])
    assert np.isfinite(out["winkler_95"])
    for tag in ("50", "80"):
        assert np.isnan(out[f"coverage_{tag}"]), f"coverage_{tag} must be N/A"
        assert np.isnan(out[f"winkler_{tag}"]), f"winkler_{tag} must be N/A"


def test_non_conformal_still_reports_all_three_levels():
    D, E, obs_D, obs_E = _one_world(52)
    out = _cell("PLC", "native", D, E, obs_D, obs_E, 52, n_samples=200)
    for tag in ("50", "80", "95"):
        assert np.isfinite(out[f"coverage_{tag}"])
        assert np.isfinite(out[f"winkler_{tag}"])


# ---------------------------------------------------------------------------
# 4 - PIT uniformity reports a caveated p-value alongside the statistic
# ---------------------------------------------------------------------------

def test_pit_reports_statistic_and_pvalue():
    D, E, obs_D, obs_E = _one_world(61)
    out = _cell("PLC", "native", D, E, obs_D, obs_E, 61, n_samples=200)
    assert 0.0 <= out["pit_ks_stat"] <= 1.0
    assert 0.0 <= out["pit_ks_pvalue"] <= 1.0
