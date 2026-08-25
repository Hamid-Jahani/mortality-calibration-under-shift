"""Validation of the semiparametric Poisson bootstrap wrapper (Brouhns et al.
2005) against synthetic truth from a KNOWN Poisson Lee-Carter process — the
PREREGISTRATION.md validation-gate pattern. Wrapped around the correctly-
specified model, the pooled bootstrap paths must (a) attain nominal-or-mildly-
conservative 95% coverage over many simulated worlds and (b) be at least as
wide on average as the model-native intervals: the bootstrap ADDS parameter
(estimation) uncertainty on top of drift + innovation noise, it can never
remove any."""
from functools import lru_cache

import numpy as np

from mortcal.eval.scores import interval_coverage
from mortcal.models.lc import LeeCarterSVD, PoissonLeeCarter
from mortcal.uq.bootstrap import PoissonBootstrap

N_AGES, T_TRAIN, H = 40, 60, 5
N_WORLDS = 10
B_TEST, N_INNER = 30, 10
N_PATHS = B_TEST * N_INNER          # classical pooled scheme: B * n_inner paths


def simulate_plc(rng):
    """One world from a known Poisson-LC DGP (same family as
    tests/test_synthetic_calibration.py); returns train (D, E) + true future mx."""
    ages = np.arange(N_AGES)
    alpha = -7.5 + 5.5 * (ages / N_AGES) ** 1.3          # log-mortality level
    beta = np.exp(-0.5 * ((ages - 12) / 14.0) ** 2)      # improvement loading
    beta = beta / beta.sum()
    mu, sigma = -1.2, 0.9                                # k_t drift, innovation sd
    k = np.cumsum(mu + rng.normal(0, sigma, T_TRAIN + H))
    k = k - k[:T_TRAIN].mean()
    mx_all = np.exp(alpha[:, None] + np.outer(beta, k))  # [ages, T+H]
    E = np.full((N_AGES, T_TRAIN + H), 1e5)
    D = rng.poisson(E * mx_all).astype(float)
    return (D[:, :T_TRAIN], E[:, :T_TRAIN]), mx_all[:, T_TRAIN:].T  # true mx [H, ages]


# ---------------------------------------------------------------------------
# fitted_mx contract (the hook the bootstrap resamples from)
# ---------------------------------------------------------------------------

def test_fitted_mx_matches_parameter_surface():
    """fitted_mx must equal exp(alpha + outer(beta, kappa)) for both LC variants
    and track observed in-sample rates closely on a correctly-specified world."""
    (D, E), _ = simulate_plc(np.random.default_rng(1))
    for cls in (LeeCarterSVD, PoissonLeeCarter):
        m = cls().fit(D, E)
        fm = m.fitted_mx()
        assert fm.shape == (N_AGES, T_TRAIN)
        np.testing.assert_allclose(
            fm, np.exp(m.alpha[:, None] + np.outer(m.beta, m.kappa)), rtol=1e-12)
        rel = np.abs(fm - D / E) / (D / E)
        assert np.median(rel) < 0.05, f"{cls.__name__} in-sample fit off: {np.median(rel):.3f}"


# ---------------------------------------------------------------------------
# wrapper interface conformance (methodology rule 4)
# ---------------------------------------------------------------------------

def test_wrapper_interface_shapes_and_positivity():
    (D, E), _ = simulate_plc(np.random.default_rng(2))
    wrap = PoissonBootstrap(PoissonLeeCarter, B=5, n_inner=2).fit(D, E)
    assert len(wrap.refits) == 5
    s = wrap.sample_mx(H, 37, np.random.default_rng(3))   # n not a multiple of B
    assert s.shape == (37, H, N_AGES)
    assert np.all(np.isfinite(s)) and np.all(s > 0)


def test_wrapper_forwards_model_kwargs_and_is_deterministic():
    """model_kwargs reach every refit; fixed rngs make the whole chain replayable
    (methodology rule 7: seeds recorded)."""
    (D, E), _ = simulate_plc(np.random.default_rng(4))
    def draw():
        w = PoissonBootstrap(PoissonLeeCarter, B=4, n_inner=2, max_iter=300).fit(
            D, E, rng=np.random.default_rng(11))
        assert all(r.max_iter == 300 for r in w.refits)
        return w.sample_mx(H, 20, np.random.default_rng(12))
    np.testing.assert_array_equal(draw(), draw())


# ---------------------------------------------------------------------------
# calibration gates: coverage and width vs model-native
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _run_worlds():
    """Shared Monte-Carlo run: per world, one bootstrap fit (B=30) and the
    matching model-native forecast from the SAME base fit (wrap.base), so the
    width comparison isolates the UQ mechanism, not refitting noise."""
    rng = np.random.default_rng(20260825)
    cov, w_boot, w_nat = [], [], []
    for _ in range(N_WORLDS):
        (D, E), true_mx = simulate_plc(rng)
        wrap = PoissonBootstrap(PoissonLeeCarter, B=B_TEST, n_inner=N_INNER).fit(
            D, E, rng=rng)
        truth = np.log(true_mx)                                   # [H, ages]
        boot = np.log(wrap.sample_mx(H, N_PATHS, rng))            # [n, H, ages]
        native = np.log(wrap.base.sample_mx(H, N_PATHS, rng))
        c, wb = interval_coverage(boot, truth, 0.95)
        _, wn = interval_coverage(native, truth, 0.95)
        cov.append(np.mean(c))
        w_boot.append(np.mean(wb))
        w_nat.append(np.mean(wn))
    return float(np.mean(cov)), float(np.mean(w_boot)), float(np.mean(w_nat))


def test_bootstrap_coverage_nominal_over_worlds():
    """Correctly-specified model + Poisson bootstrap: pooled 95% intervals must
    cover at (or conservatively above) nominal across >= 10 worlds."""
    coverage, _, _ = _run_worlds()
    assert 0.90 <= coverage <= 0.995, f"bootstrap 95% attained {coverage:.3f}"


def test_bootstrap_at_least_as_wide_as_model_native():
    """Bootstrap = native forecast noise + parameter uncertainty, so its mean
    interval width can only exceed (or match) the native mechanism's."""
    _, w_boot, w_nat = _run_worlds()
    assert w_boot >= w_nat, (
        f"bootstrap width {w_boot:.4f} < native width {w_nat:.4f} — "
        "parameter uncertainty vanished")
