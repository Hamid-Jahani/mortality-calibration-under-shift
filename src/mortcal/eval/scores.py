"""Proper scoring rules and calibration diagnostics.

All scoring functions share one convention: forecasts arrive as SAMPLES from the
predictive distribution (shape [n_samples, ...]), truths as arrays broadcastable
to the trailing dims. This is deliberate: rule 4 of the methodology — every model,
classical or neural, emits samples through one interface, so one evaluation code
path serves all of them and no model gets a bespoke (and accidentally flattering)
metric implementation.

Sign convention: scores are negatively oriented (smaller = better) throughout.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def crps_sample(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """CRPS estimated from predictive samples (Gneiting & Raftery 2007, eq. 21).

    CRPS(F, y) = E|X - y| - 0.5 E|X - X'|   with X, X' ~ F independent.

    samples: [m, *dims], truth: [*dims]. Returns [*dims].
    O(m log m) estimator: E|X - X'| computed from the sorted samples via the
    Gini identity  sum_{i,j} |s_i - s_j| = 2 * sum_i (2i - m + 1) s_i  (0-indexed),
    divided by m^2 (the unbiased-in-m plug-in form matching eq. 21's plug-in F).
    """
    m = samples.shape[0]
    s = np.sort(samples, axis=0)
    term1 = np.mean(np.abs(s - truth[None, ...]), axis=0)
    w = (2.0 * np.arange(m) - m + 1.0).reshape((m,) + (1,) * (samples.ndim - 1))
    e_xx = 2.0 * np.sum(w * s, axis=0) / (m * m)
    return term1 - 0.5 * e_xx


def log_score_poisson(lam: np.ndarray, deaths: np.ndarray) -> np.ndarray:
    """Negative Poisson log predictive density for observed death counts.

    lam: predictive Poisson means [m, *dims] (mixture over m samples) or [*dims].
    Mixture handled by log-mean-exp over the sample axis.
    """
    lam = np.asarray(lam, dtype=float)
    if lam.ndim == deaths.ndim:
        return -stats.poisson.logpmf(np.round(deaths), lam)
    logp = stats.poisson.logpmf(np.round(deaths)[None, ...], lam)  # [m, *dims]
    m = logp.shape[0]
    return -(np.logaddexp.reduce(logp, axis=0) - np.log(m))


def interval_coverage(samples: np.ndarray, truth: np.ndarray, level: float = 0.95):
    """Empirical coverage of the central `level` interval + mean width.

    Returns (covered: bool array [*dims], width: array [*dims]).
    Report BOTH — coverage without width is how PICP=1.0 gets celebrated.
    """
    alpha = 1.0 - level
    lo = np.quantile(samples, alpha / 2, axis=0)
    hi = np.quantile(samples, 1 - alpha / 2, axis=0)
    covered = (truth >= lo) & (truth <= hi)
    return covered, hi - lo


def winkler_score(samples: np.ndarray, truth: np.ndarray, level: float = 0.95) -> np.ndarray:
    """Winkler/interval score (Gneiting & Raftery 2007, eq. 43). Negatively oriented.

    Width plus 2/alpha penalty per unit of miss — the width-penalised criterion
    whose absence from the mortality literature GAP-ANALYSIS.md documents.
    """
    alpha = 1.0 - level
    lo = np.quantile(samples, alpha / 2, axis=0)
    hi = np.quantile(samples, 1 - alpha / 2, axis=0)
    below = (lo - truth) * (truth < lo)
    above = (truth - hi) * (truth > hi)
    return (hi - lo) + (2.0 / alpha) * (below + above)


def pit_values(samples: np.ndarray, truth: np.ndarray, randomize: bool = True,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Probability integral transform of truth under the empirical predictive CDF.

    Uniform[0,1] iff calibrated. `randomize` applies the randomised PIT
    (Czado, Gneiting & Held 2009) so discrete/tied predictive samples still
    yield exact uniformity under the null.
    """
    m = samples.shape[0]
    below = np.sum(samples < truth[None, ...], axis=0).astype(float)
    equal = np.sum(samples == truth[None, ...], axis=0).astype(float)
    if randomize:
        rng = rng or np.random.default_rng(0)
        v = rng.uniform(size=truth.shape)
    else:
        v = 0.5
    return (below + v * (equal + 1.0)) / (m + 1.0)


def joint_path_coverage(samples: np.ndarray, truth: np.ndarray, level: float = 0.95) -> float:
    """Fraction of forecast PATHS entirely inside their pointwise `level` bands.

    samples: [m, H, *units], truth: [H, *units]. A path (one unit's h=1..H
    trajectory) counts as covered only if EVERY horizon is inside its band.
    The gap between this and pointwise marginal coverage is hypothesis H3.
    """
    covered, _ = interval_coverage(samples, truth, level)  # [H, *units]
    return float(np.mean(np.all(covered, axis=0)))
