"""The GP's closed-form median (posterior mean) must agree with the sampled
median the conformal wrappers previously estimated — and the wrappers must
actually take the shortcut, so the 1.59 GB solo OOM of GP/split_conf cannot
recur. Small synthetic panel; gated tests for the GP itself live in
tests/test_neural.py."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gpytorch")

from mortcal.models.gp import MultiOutputGP           # noqa: E402
from mortcal.uq.conformal import _median_log_forecast, SplitConformalMx  # noqa: E402

N_AGES, T = 12, 48


def _panel(rng):
    ages = np.arange(N_AGES)
    alpha = -6.5 + 3.5 * ages / N_AGES
    beta = np.full(N_AGES, 1.0 / N_AGES)
    k = np.cumsum(-1.0 + rng.normal(0, 0.6, T))
    mx = np.exp(alpha[:, None] + np.outer(beta, k))
    E = np.full((N_AGES, T), 2e5)
    return rng.poisson(E * mx).astype(float), E


@pytest.fixture(scope="module")
def gp():
    rng = np.random.default_rng(3)
    D, E = _panel(rng)
    return MultiOutputGP(min_years=30).fit(D, E), D, E


def test_median_logmx_matches_sampled_median(gp):
    model, _, _ = gp
    h = 4
    closed = model.median_logmx(h)
    rng = np.random.default_rng(0)
    sampled = np.median(np.log(model.sample_mx(h, 400, rng)), axis=0)
    assert closed.shape == (h, N_AGES)
    assert np.all(np.isfinite(closed))
    # Gaussian posterior: sampled median -> posterior mean; MC error on 400 draws
    assert np.abs(closed - sampled).max() < 0.05 * np.abs(closed).mean() + 0.05


def test_conformal_center_takes_the_shortcut(gp):
    model, D, E = gp
    calls = {"sample_mx": 0}
    orig = model.sample_mx

    def spy(h, n, rng):
        calls["sample_mx"] += 1
        return orig(h, n, rng)

    model.sample_mx = spy
    med = _median_log_forecast(model, 3, 1000, np.random.default_rng(1))
    assert calls["sample_mx"] == 0, "wrapper sampled the GP instead of using median_logmx"
    assert med.shape == (3, N_AGES)


def test_split_conformal_on_gp_fits_without_sampling_the_posterior(gp):
    _, D, E = gp
    w = SplitConformalMx(lambda: MultiOutputGP(min_years=30), alpha=0.1,
                         cal_years=6, n_median_samples=10_000_000)  # would OOM if sampled
    w.fit(D, E)
    s = w.sample_mx(3, 50, np.random.default_rng(2))
    assert s.shape == (50, 3, N_AGES) and np.all(np.isfinite(s))
