"""Validation of the banded sparse VAR (svar.py) against a KNOWN banded-VAR
truth (methodology rule 5): the DGP is exactly the model class — a banded
VAR(1) on Delta log m with Gaussian innovations — so the fitted model must
(a) recover the band coefficients and (b) attain nominal interval coverage on
future log m. If either fails, svar results on real data are untrustworthy."""
import numpy as np

from mortcal.eval import interval_coverage
from mortcal.models.svar import SparseVAR

N_AGES, T_TRAIN, H = 24, 120, 8
W_TRUE = 3                       # true band == model band W=3 (correct spec)
N_SAMPLES, N_WORLDS = 800, 20


def _true_params():
    """Stable banded VAR(1): Gershgorin row sums 0.68 < 1; ~2%/yr improvement."""
    A = np.zeros((N_AGES, N_AGES))
    weights = {0: 0.30, 1: 0.12, 2: 0.05, 3: 0.02}
    for d, wgt in weights.items():
        A += wgt * np.eye(N_AGES, k=d)
        if d:
            A += wgt * np.eye(N_AGES, k=-d)
    c = np.full(N_AGES, -0.006)              # (I-A)^-1 c ~ -0.02 mean improvement
    ii = np.arange(N_AGES)
    sigma = 0.02 * (0.3 ** np.abs(ii[:, None] - ii[None, :]))  # correlated innovations
    return A, c, sigma


def simulate_world(rng, h=H):
    """Train (D, E) from the banded-VAR truth + one true future log m path."""
    A, c, sigma = _true_params()
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
