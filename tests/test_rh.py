"""Validation of the Renshaw-Haberman M2-A model against synthetic truth from
a KNOWN M2-A process (pattern of tests/test_synthetic_calibration.py): the DGP
is exactly log mu = a_x + b_x k_t + (1/n_ages) g_{t-x} with Poisson deaths, so
the fitted model is correctly specified and must (i) recover the latent
indices, (ii) converge, and (iii) attain near-nominal interval coverage on the
true future rates. If any gate fails, real-data RH results measure a bug."""
import numpy as np
import pytest

from mortcal.eval import interval_coverage
from mortcal.models.rh import RenshawHaberman
from mortcal.uq.bootstrap import PoissonBootstrap

N_AGES, T_TRAIN, H, N_SAMPLES, N_WORLDS = 30, 60, 5, 1000, 15
B2 = 1.0 / N_AGES


def simulate_m2a(rng, exposure=1e6):
    """One world from a known M2-A DGP.

    Returns train (D, E), true future mx [H, ages], true k (T+H), true g
    (all cohorts, oldest first — same indexing as RenshawHaberman.gamma).
    """
    ages = np.arange(N_AGES)
    alpha = -7.8 + 5.2 * (ages / N_AGES) ** 1.2          # log-level by age
    beta = np.exp(-0.5 * ((ages - 8) / 12.0) ** 2)       # period loading
    beta = beta / beta.sum()
    k = np.cumsum(-1.0 + rng.normal(0, 0.8, T_TRAIN + H))
    k = k - k[:T_TRAIN].mean()
    n_coh = N_AGES - 1 + T_TRAIN + H
    phi_g, sig_g = 0.7, 0.5                              # stationary AR(1) cohort index
    g = np.empty(n_coh)
    g[0] = rng.normal(0.0, sig_g / np.sqrt(1 - phi_g ** 2))
    for c in range(1, n_coh):
        g[c] = phi_g * g[c - 1] + rng.normal(0.0, sig_g)
    cidx = np.arange(T_TRAIN + H)[None, :] - ages[:, None] + (N_AGES - 1)
    mx_all = np.exp(alpha[:, None] + np.outer(beta, k) + B2 * g[cidx])
    E = np.full((N_AGES, T_TRAIN + H), exposure)
    D = rng.poisson(E * mx_all).astype(float)
    return (D[:, :T_TRAIN], E[:, :T_TRAIN]), mx_all[:, T_TRAIN:].T, k, g


@pytest.fixture(scope="module")
def fitted_world():
    rng = np.random.default_rng(20260825)
    (D, E), true_mx, k_true, g_true = simulate_m2a(rng)
    model = RenshawHaberman().fit(D, E)
    return model, true_mx, k_true, g_true


def test_converged_on_synthetic_data(fitted_world):
    """RH pathologies are a known schedule risk — the damped alternating-Newton
    fit must actually converge (not just stop) on clean synthetic data."""
    model, *_ = fitted_world
    assert model.converged
    assert model.n_iter < model.max_iter


def test_kappa_recovery(fitted_world):
    model, _, k_true, _ = fitted_world
    r = np.corrcoef(model.kappa, k_true[:T_TRAIN])[0, 1]
    assert r > 0.98, f"kappa correlation {r:.4f}"


def test_cohort_index_recovery(fitted_world):
    """Compare on well-observed (retained) cohorts only, after moving the TRUE
    gamma into the same gauge constraint 3 imposes on the fitted gamma —
    {1, c, c^2} removed (level/trend are gauge freedoms, the quadratic is the
    pinned near-invariance) — so only the identified shape is compared."""
    model, _, _, g_true = fitted_world
    ret = model.gamma_retained
    c = model.cohort_index
    n_c = len(ret)
    gt = g_true[:n_c]
    cidx = np.arange(T_TRAIN)[None, :] - np.arange(N_AGES)[:, None] + (N_AGES - 1)
    counts = np.bincount(cidx.ravel(), minlength=n_c)

    def corr(sel):
        coef = np.polyfit(c[sel], gt[sel], 2)
        return np.corrcoef(model.gamma[sel], gt[sel] - np.polyval(coef, c[sel]))[0, 1]

    # Edge cohorts retained with 5-14 observations have se(gamma_c) comparable
    # to sd(g) itself (b2 = 1/n_ages shrinks the per-cell signal), so raw
    # correlation there measures noise, not the estimator. Gate strictly where
    # information exists; keep a loose all-retained floor to catch gross bugs.
    r15 = corr(ret & (counts >= 15))
    assert r15 > 0.95, f"well-observed cohort correlation {r15:.4f}"
    r_all = corr(ret)
    assert r_all > 0.8, f"all-retained cohort correlation {r_all:.4f}"


def test_identifiability_constraints_hold(fitted_world):
    model, *_ = fitted_world
    assert np.isclose(model.beta.sum(), 1.0)
    assert np.isclose(model.kappa.sum(), 0.0, atol=1e-8)
    ret = model.gamma_retained
    c = model.cohort_index
    t1, t0 = np.polyfit(c[ret], model.gamma[ret], 1)
    assert abs(t0) < 1e-6 and abs(t1) < 1e-6, "retained gamma not detrended"


def test_sample_mx_shape_and_finiteness(fitted_world):
    model, *_ = fitted_world
    rng = np.random.default_rng(1)
    s = model.sample_mx(H, 64, rng)
    assert s.shape == (64, H, N_AGES)
    assert np.all(np.isfinite(s)) and np.all(s > 0)


def test_nominal_coverage_over_worlds():
    """Correctly-specified M2-A over >= 15 independent worlds: 95% intervals on
    log true future rates must land near nominal. Band [0.88, 0.99] allows for
    unpropagated (a, b, g)-estimation noise (a model-native-UQ property, as in
    lc.py) plus Monte-Carlo error, while still catching real miscalibration."""
    rng = np.random.default_rng(42)
    covs, conv = [], []
    for _ in range(N_WORLDS):
        (D, E), true_mx, _, _ = simulate_m2a(rng)
        model = RenshawHaberman().fit(D, E)
        conv.append(model.converged)
        samples = np.log(model.sample_mx(H, N_SAMPLES, rng))    # [n, H, ages]
        cov, _ = interval_coverage(samples, np.log(true_mx), 0.95)
        covs.append(cov.mean())
    assert all(conv), f"fits failed to converge in {conv.count(False)} worlds"
    coverage = float(np.mean(covs))
    assert 0.88 <= coverage <= 0.99, f"nominal 95% attained {coverage:.3f}"


# ---------------------------------------------------------------------------
# fitted_mx contract (the Poisson-bootstrap hook)
# ---------------------------------------------------------------------------

def test_fitted_mx_surface_covers_excluded_cohorts_with_imputed_gamma():
    """fitted_mx = exp(a + b k + b2 g_{t-x}) on the whole rectangle, finite and
    positive. Cells of cohorts EXCLUDED from estimation (the sparse corner
    diagonals, weight 0 in the fit) must read the IMPUTED gamma — the linear
    trend of the retained cohorts — so the Poisson bootstrap can resample
    every cell. Against observed rates the median relative error is < 5%."""
    rng = np.random.default_rng(3)
    (D, E), *_ = simulate_m2a(rng)
    model = RenshawHaberman().fit(D, E)
    assert model.converged
    fm = model.fitted_mx()
    assert fm.shape == (N_AGES, T_TRAIN)
    assert np.all(np.isfinite(fm)) and np.all(fm > 0)

    cidx = np.arange(T_TRAIN)[None, :] - np.arange(N_AGES)[:, None] + (N_AGES - 1)
    eta = model.alpha[:, None] + np.outer(model.beta, model.kappa) + B2 * model.gamma[cidx]
    np.testing.assert_allclose(fm, np.exp(eta), rtol=1e-12)

    excluded = ~model.gamma_retained[cidx]
    assert excluded.any(), "DGP has no sparse cohorts; the imputation path is untested"
    ret, c = model.gamma_retained, model.cohort_index
    tr1, tr0 = np.polyfit(c[ret], model.gamma[ret], 1)
    g_imputed = tr0 + tr1 * c
    eta_imp = (model.alpha[:, None] + np.outer(model.beta, model.kappa)
               + B2 * g_imputed[cidx])
    np.testing.assert_allclose(fm[excluded], np.exp(eta_imp)[excluded], rtol=1e-12)

    rel = np.abs(fm - D / E) / (D / E)
    assert np.median(rel) < 0.05, f"in-sample fit off: {np.median(rel):.3f}"


def test_poisson_bootstrap_rh_end_to_end():
    """PoissonBootstrap around RH: base fitted_mx resamplable at every cell,
    every refit converges on its Poisson pseudo-panel, pooled paths have the
    study-wide shape and are finite."""
    rng = np.random.default_rng(11)
    (D, E), *_ = simulate_m2a(rng)
    wrap = PoissonBootstrap(RenshawHaberman, B=10, n_inner=2).fit(
        D, E, rng=np.random.default_rng(12))
    assert len(wrap.refits) == 10
    assert all(r.converged for r in wrap.refits)
    s = wrap.sample_mx(H, 23, np.random.default_rng(13))
    assert s.shape == (23, H, N_AGES)
    assert np.all(np.isfinite(s)) and np.all(s > 0)
