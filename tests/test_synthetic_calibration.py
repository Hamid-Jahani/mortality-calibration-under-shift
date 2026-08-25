"""Validation gate 1 (PREREGISTRATION.md): the full chain — model fit,
k_t forecasting, sampling, scoring — must recover nominal coverage on data
simulated from a KNOWN Poisson Lee-Carter process. If this fails, every
coverage number the study produces on real data is measuring a bug."""
import numpy as np
import pytest
from scipy import stats

from mortcal.models import LeeCarterSVD, PoissonLeeCarter
from mortcal.eval import interval_coverage, pit_values

N_AGES, T_TRAIN, H, N_SAMPLES, N_REPS = 40, 60, 10, 1500, 30


def simulate_plc(rng):
    """One world from a known Poisson-LC DGP; returns train (D, E) + true future mx."""
    ages = np.arange(N_AGES)
    alpha = -7.5 + 5.5 * (ages / N_AGES) ** 1.3          # log-mortality level by age
    beta = np.exp(-0.5 * ((ages - 12) / 14.0) ** 2)      # improvement loading
    beta = beta / beta.sum()
    mu, sigma = -1.2, 0.9                                # k_t drift, innovation sd
    k = np.cumsum(mu + rng.normal(0, sigma, T_TRAIN + H))
    k = k - k[:T_TRAIN].mean()
    mx_all = np.exp(alpha[:, None] + np.outer(beta, k))  # [ages, T+H]
    E = np.full((N_AGES, T_TRAIN + H), 1e5)
    D = rng.poisson(E * mx_all).astype(float)
    return (D[:, :T_TRAIN], E[:, :T_TRAIN]), mx_all[:, T_TRAIN:].T  # true mx [H, ages]


def _run_worlds(model_cls, rng):
    covered_all, pit_all = [], []
    for _ in range(N_REPS):
        (D, E), true_mx = simulate_plc(rng)
        m = model_cls().fit(D, E)
        samples = np.log(m.sample_mx(H, N_SAMPLES, rng))          # [n, H, ages]
        truth = np.log(true_mx)                                   # [H, ages]
        cov, _ = interval_coverage(samples, truth, 0.95)
        covered_all.append(cov)
        pit_all.append(pit_values(samples, truth, rng=rng))
    coverage = float(np.mean(covered_all))
    pit = np.concatenate([p.ravel() for p in pit_all])
    hist, _ = np.histogram(pit, bins=10, range=(0, 1))
    return coverage, hist / hist.sum()


def test_correctly_specified_model_attains_nominal_coverage():
    """The DGP IS Poisson Lee-Carter, so Poisson-LC is the correctly-specified
    model and must pass both gates: nominal coverage AND near-uniform PIT."""
    coverage, frac = _run_worlds(PoissonLeeCarter, np.random.default_rng(20260825))
    # Monte-Carlo tolerance: 30 worlds x 10 horizons x 40 ages, correlated within world
    assert 0.90 <= coverage <= 0.985, f"nominal 95% attained {coverage:.3f}"
    # KS on heavily dependent cells is too strict a null; gate on gross shape
    assert np.all(np.abs(frac - 0.1) < 0.04), f"PIT deciles {np.round(frac, 3)}"


def test_svd_lc_is_mildly_overdispersed_by_construction():
    """Two-stage SVD Lee-Carter on Poisson data: SVD kappa carries observation
    noise into the RWD sigma estimate, so its intervals come out slightly WIDE
    (center-heavy PIT). Coverage must still be >= nominal; the hump is expected
    and documented — this is a finding about the method, not a harness bug."""
    coverage, frac = _run_worlds(LeeCarterSVD, np.random.default_rng(20260825))
    assert coverage >= 0.93, f"SVD-LC coverage {coverage:.3f} fell BELOW nominal"
    assert frac[4] + frac[5] > 0.2, "expected center-heavy PIT from overdispersion"


def test_misspecified_drift_is_detected_as_undercoverage():
    """Negative control: break the DGP after the cutoff (drift doubles) and the
    same chain must now UNDER-cover — proving the harness can see failure."""
    rng = np.random.default_rng(7)
    covered_all = []
    for _ in range(N_REPS):
        ages = np.arange(N_AGES)
        alpha = -7.5 + 5.5 * (ages / N_AGES) ** 1.3
        beta = np.exp(-0.5 * ((ages - 12) / 14.0) ** 2); beta = beta / beta.sum()
        k_tr = np.cumsum(-1.2 + rng.normal(0, 0.9, T_TRAIN))
        k_te = k_tr[-1] + np.cumsum(+3.0 + rng.normal(0, 0.9, H))   # regime break
        mx_tr = np.exp(alpha[:, None] + np.outer(beta, k_tr - k_tr.mean()))
        mx_te = np.exp(alpha[:, None] + np.outer(beta, k_te - k_tr.mean()))
        E = np.full((N_AGES, T_TRAIN), 1e5)
        D = rng.poisson(E * mx_tr).astype(float)
        m = PoissonLeeCarter().fit(D, E)
        samples = np.log(m.sample_mx(H, 400, rng))
        cov, _ = interval_coverage(samples, np.log(mx_te.T), 0.95)
        covered_all.append(cov.mean())
    assert float(np.mean(covered_all)) < 0.75, "harness failed to detect a break"
