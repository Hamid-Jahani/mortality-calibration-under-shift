"""Validation of the banded sparse VAR (svar.py) against a KNOWN banded-VAR
truth (methodology rule 5): the DGP is exactly the model class — a banded
VAR(1) on Delta log m with Gaussian innovations — so the fitted model must
(a) recover the band coefficients and (b) attain nominal interval coverage on
future log m. If either fails, svar results on real data are untrustworthy."""
import numpy as np
import pytest

from mortcal.eval import interval_coverage
from mortcal.models.svar import SparseVAR
from mortcal.uq.bootstrap import PoissonBootstrap

N_AGES, T_TRAIN, H = 24, 120, 8
W_TRUE = 3                       # true band == model band W=3 (correct spec)
N_SAMPLES, N_WORLDS = 800, 20


def _true_params(innov_var=0.02):
    """Stable banded VAR(1): Gershgorin row sums 0.68 < 1; ~2%/yr improvement.
    innov_var is the per-age innovation VARIANCE (default 0.02, sd 0.14)."""
    A = np.zeros((N_AGES, N_AGES))
    weights = {0: 0.30, 1: 0.12, 2: 0.05, 3: 0.02}
    for d, wgt in weights.items():
        A += wgt * np.eye(N_AGES, k=d)
        if d:
            A += wgt * np.eye(N_AGES, k=-d)
    c = np.full(N_AGES, -0.006)              # (I-A)^-1 c ~ -0.02 mean improvement
    ii = np.arange(N_AGES)
    sigma = innov_var * (0.3 ** np.abs(ii[:, None] - ii[None, :]))  # correlated innovations
    return A, c, sigma


def simulate_world(rng, h=H, innov_var=0.02):
    """Train (D, E) from the banded-VAR truth + one true future log m path."""
    A, c, sigma = _true_params(innov_var)
    L = np.linalg.cholesky(sigma)
    y = np.zeros(N_AGES)
    for _ in range(50):                                  # burn-in toward stationarity
        y = c + A @ y + L @ rng.standard_normal(N_AGES)
    ages = np.arange(N_AGES)
    logm = np.empty((N_AGES, T_TRAIN))
    logm[:, 0] = -8.0 + 5.0 * (ages / N_AGES) ** 1.2     # level: VAR only sees diffs
    for t in range(1, T_TRAIN):
        y = c + A @ y + L @ rng.standard_normal(N_AGES)
        logm[:, t] = logm[:, t - 1] + y
    future = np.empty((h, N_AGES))
    level = logm[:, -1].copy()
    for t in range(h):
        y = c + A @ y + L @ rng.standard_normal(N_AGES)
        level = level + y
        future[t] = level
    E = np.full((N_AGES, T_TRAIN), 1e5)
    D = E * np.exp(logm)          # exact rates: the KNOWN process is the VAR itself
    return (D, E), future         # future: true log m [h, ages]


def test_coefficient_recovery_on_band():
    """OLS on the correctly-banded design must recover the true coefficients:
    mean absolute error over the band entries of A below 0.1."""
    (D, E), _ = simulate_world(np.random.default_rng(42))
    m = SparseVAR(W=3).fit(D, E)
    A_true, c_true, _ = _true_params()
    ii = np.arange(N_AGES)
    band = np.abs(ii[:, None] - ii[None, :]) <= W_TRUE
    mae = float(np.mean(np.abs(m.A_[band] - A_true[band])))
    assert mae < 0.1, f"band coefficient MAE {mae:.3f}"
    assert float(np.mean(np.abs(m.c_ - c_true))) < 0.02, "intercepts off"


def test_off_band_is_exactly_zero():
    """The band IS the sparsity: everything outside |i-j| <= W must be 0."""
    (D, E), _ = simulate_world(np.random.default_rng(1))
    m = SparseVAR(W=3).fit(D, E)
    ii = np.arange(N_AGES)
    off = np.abs(ii[:, None] - ii[None, :]) > 3
    assert np.all(m.A_[off] == 0.0)


def test_sample_shape_and_finiteness():
    (D, E), _ = simulate_world(np.random.default_rng(2))
    m = SparseVAR().fit(D, E)
    s = m.sample_mx(H, 64, np.random.default_rng(3))
    assert s.shape == (64, H, N_AGES)
    assert np.all(np.isfinite(s)) and np.all(s > 0)


def test_nominal_coverage_on_banded_var_truth():
    """Correctly-specified model on its own DGP: average 95% marginal coverage
    of future log m over >= 20 worlds must land in [0.88, 0.99]."""
    rng = np.random.default_rng(20260825)
    covered = []
    for _ in range(N_WORLDS):
        (D, E), true_logm = simulate_world(rng)
        m = SparseVAR().fit(D, E)
        samples = np.log(m.sample_mx(H, N_SAMPLES, rng))     # [n, H, ages]
        cov, _ = interval_coverage(samples, true_logm, 0.95)
        covered.append(cov.mean())
    coverage = float(np.mean(covered))
    assert 0.88 <= coverage <= 0.99, f"coverage {coverage:.3f}"


# ---------------------------------------------------------------------------
# fitted_mx contract (the Poisson-bootstrap hook)
# ---------------------------------------------------------------------------

def _fitted_world():
    """Low-noise world (innovation sd 0.02) for in-sample closeness checks.
    fitted - observed IS the innovation for a VAR, so the default DGP's sd of
    0.14 would put even a perfect one-step fit ~10% off in median."""
    (D, E), _ = simulate_world(np.random.default_rng(5), innov_var=4e-4)
    return D, E, SparseVAR().fit(D, E)


def test_fitted_mx_one_step_tracks_observed_rates():
    """Default fitted_mx is the one-step-ahead conditional mean
    exp(log m[t-1] + c + A y[t-1]): panel shape, finite, observed rates in
    years 0-1 (no lag to regress on), and median relative error < 5%."""
    D, E, m = _fitted_world()
    mx = D / E
    fm = m.fitted_mx()
    assert fm.shape == (N_AGES, T_TRAIN)
    assert np.all(np.isfinite(fm)) and np.all(fm > 0)
    np.testing.assert_allclose(fm[:, :2], mx[:, :2], rtol=1e-12)
    logm = np.log(mx)
    Y = np.diff(logm, axis=1)
    expect = logm[:, 1:-1] + m.c_[:, None] + m.A_ @ Y[:, :-1]
    np.testing.assert_allclose(np.log(fm[:, 2:]), expect, rtol=0, atol=1e-10)
    rel = np.abs(fm - mx) / mx
    assert np.median(rel) < 0.05, f"one-step fit off: {np.median(rel):.3f}"


def test_fitted_mx_cumulative_pinned_at_both_ends_but_drifts():
    """The cumulative reconstruction is exact in years 0-1 (observed) and in
    the last year (OLS-with-intercept residuals sum to zero), yet drifts by
    the cumulated residuals in between: its median error must exceed the
    one-step surface's — the reason one_step is the bootstrap default."""
    D, E, m = _fitted_world()
    mx = D / E
    fc = m.fitted_mx(how="cumulative")
    assert fc.shape == (N_AGES, T_TRAIN)
    assert np.all(np.isfinite(fc)) and np.all(fc > 0)
    np.testing.assert_allclose(fc[:, :2], mx[:, :2], rtol=1e-12)
    np.testing.assert_allclose(fc[:, -1], mx[:, -1], rtol=1e-8)
    rel_cum = np.median(np.abs(fc - mx) / mx)
    rel_one = np.median(np.abs(m.fitted_mx() - mx) / mx)
    assert rel_cum > rel_one, f"cumulative {rel_cum:.3f} vs one-step {rel_one:.3f}"
    with pytest.raises(ValueError):
        m.fitted_mx(how="bogus")


def test_poisson_bootstrap_svar_end_to_end():
    """PoissonBootstrap around the banded VAR: pooled paths keep the
    study-wide shape and are finite and positive."""
    (D, E), _ = simulate_world(np.random.default_rng(6))
    wrap = PoissonBootstrap(SparseVAR, B=10, n_inner=2).fit(
        D, E, rng=np.random.default_rng(7))
    assert len(wrap.refits) == 10
    s = wrap.sample_mx(H, 23, np.random.default_rng(8))
    assert s.shape == (23, H, N_AGES)
    assert np.all(np.isfinite(s)) and np.all(s > 0)
