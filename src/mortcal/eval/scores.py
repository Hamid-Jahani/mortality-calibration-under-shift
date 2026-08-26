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


def round_deaths(deaths: np.ndarray) -> np.ndarray:
    """Round observed deaths to the nearest integer, halves UP: floor(d + 0.5).

    PREREGISTRATION.md fixes the Poisson log-score convention as observed
    deaths "rounded to the nearest integer": HMD death counts are fractional
    (Lexis-triangle splitting and the redistribution of deaths of unknown age,
    HMD Methods Protocol v6). Measured on the all-country ``Deaths_1x1.txt``
    (HMD, last modified 2026-06-15): 40% of cells are non-integer and 2.2%
    land on exactly .5 (0.1% inside the 2020-2024 test window). ``np.round``
    implements IEEE round-half-to-EVEN (2.5 -> 2, 3.5 -> 4), so on those cells
    the direction of the rounding would flip with the parity of the integer
    part -- a cell-dependent perturbation of the truth (1.1% of all cells)
    that no reader would infer from "nearest integer". Banker's rounding was
    therefore rejected; half-up is what the pre-registration means, and it is
    applied to every model family through this one helper.

    Deaths are non-negative, so the behaviour on negative halves (floor(d+0.5)
    sends -2.5 to -2, i.e. half toward +inf) never arises. Returns a float
    array of integer values so it broadcasts against sample arrays.
    """
    return np.floor(np.asarray(deaths, dtype=float) + 0.5)


#: Half-count continuity correction for zero-death cells (addendum 2 §2).
HALF_COUNT = 0.5


def log_crude_rate(deaths: np.ndarray, exposure: np.ndarray) -> np.ndarray:
    """Log crude death rate log(max(D, 0.5) / E) — the rate-scale convention.

    A cell with zero observed deaths has no finite log rate. The alternatives
    are to drop it, to floor the RATE, or to floor the COUNT; PREREGISTRATION
    -ADDENDUM-2 §2 registers the last of these, the standard demographic
    half-count continuity correction, and forbids dropping.

    Flooring the rate is what this function replaces, and it was not a
    cosmetic choice. With the old 1e-10 rate floor a zero-death cell scored
    as log(1e-10) = -23.03 instead of log(0.5/E) ~ -8.3 — finite, so the age
    mask kept it, and roughly 85x the normal squared error. In the 2020-2024
    test window that is 10.4% of cells for ISL and LUX (52 of 500 each), both
    of which are in SHIFT_POPS: their rate-scale metrics would have been
    dominated by the floor rather than by mortality, and the damage would
    have concentrated at young ages and in small populations — exactly the
    pattern H4 predicts, which is how an artefact gets read as a finding.

    Applied to BOTH sides of every rate-scale score: the observed rate and
    the Poisson-inclusive predictive samples share this convention, so the
    correction cancels rather than biasing the comparison.
    """
    d = np.maximum(np.asarray(deaths, dtype=float), HALF_COUNT)
    return np.log(d / np.asarray(exposure, dtype=float))


def log_score_poisson(lam: np.ndarray, deaths: np.ndarray) -> np.ndarray:
    """Negative Poisson log predictive density for observed death counts.

    lam: predictive Poisson means [m, *dims] (mixture over m samples) or [*dims].
    Mixture handled by log-mean-exp over the sample axis. Observed deaths are
    integerised with ``round_deaths`` (half-up) -- the pre-registered
    convention; ``crps_counts`` is its rounding-free sensitivity companion.
    """
    lam = np.asarray(lam, dtype=float)
    deaths = np.asarray(deaths, dtype=float)
    d = round_deaths(deaths)
    if lam.ndim == deaths.ndim:
        return -stats.poisson.logpmf(d, lam)
    logp = stats.poisson.logpmf(d[None, ...], lam)  # [m, *dims]
    m = logp.shape[0]
    return -(np.logaddexp.reduce(logp, axis=0) - np.log(m))


def crps_counts(death_samples: np.ndarray, obs_deaths: np.ndarray) -> np.ndarray:
    """CRPS on the death-COUNT scale: the registered sensitivity companion to
    ``log_score_poisson`` (PREREGISTRATION.md, Metrics).

    The Poisson log score needs an integer truth, hence ``round_deaths``. CRPS
    does not: it is defined for any real truth and any predictive law, discrete
    or continuous (Gneiting & Raftery 2007, Sec. 4.2). Scoring the same sampled
    death paths against the UNROUNDED observed deaths therefore tests whether
    any count-scale conclusion hinges on the rounding convention. Nothing is
    rounded here, deliberately.

    death_samples: [m, *dims] sampled death counts -- Poisson draws
    D ~ Poi(m_x E) (the predictive law for D) or the sampled means m_x E
    (a predictive law for E[D]); state which is used when reporting.
    obs_deaths: [*dims]. Returns [*dims] via the ``crps_sample`` estimator.
    """
    return crps_sample(np.asarray(death_samples, dtype=float),
                       np.asarray(obs_deaths, dtype=float))


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


def _bin_unit_interval(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Index of the equal-width bin [k/K, (k+1)/K) containing x; last bin
    closed at 1 so x == 1.0 lands in bin K-1."""
    return np.minimum((x * n_bins).astype(int), n_bins - 1)


def murphy_decomposition(pit_or_probs: np.ndarray, outcomes: np.ndarray,
                         n_bins: int | None = 10) -> dict[str, float]:
    """Murphy (1973) reliability-resolution-uncertainty partition of the Brier score.

    Murphy, A. H. (1973), "A new vector partition of the probability score",
    J. Appl. Meteorol. 12, 595-600; general loss-function form in Pohle (2020),
    arXiv:2005.01835 (literature/pdf). For forecast probabilities f_i in [0, 1],
    binary outcomes o_i, N cases grouped into bins k with n_k members,
    bin-mean forecast fbar_k, bin-mean outcome obar_k and base rate obar:

        BS  = (1/N) sum_i (f_i - o_i)^2
        REL = (1/N) sum_k n_k (fbar_k - obar_k)^2   forecast vs realised frequency
        RES = (1/N) sum_k n_k (obar_k - obar)^2     realised frequency moves with f
        UNC = obar (1 - obar)                       Brier score of the base rate
        BS  = REL - RES + UNC                       exact if f is constant within bins.

    When forecasts vary inside a bin the identity acquires two extra terms
    (Stephenson, Coelho & Jolliffe 2008, Weather Forecast. 23, 752-757):
    BS = REL - RES + UNC + WBV - WBC, with within-bin variance
    WBV = (1/N) sum_i (f_i - fbar_k(i))^2 and within-bin covariance
    WBC = (2/N) sum_i (f_i - fbar_k(i)) (o_i - obar_k(i)). Both are returned so
    the identity can be checked exactly under any binning.

    Use in this study: the outcome is a cell's interval-hit indicator at one
    nominal level and the forecast is the probability the model assigned to a
    hit. Two cautions the numbers must be read with:

    * With a CONSTANT forecast (every cell gets f_i = nominal level, e.g. 0.95)
      only one bin is occupied, so REL = (nominal - empirical coverage)^2,
      RES = 0 and BS = REL + UNC. The decomposition then adds nothing beyond
      the coverage gap. That degeneracy is why ``murphy_pit`` is reported
      alongside it: it uses the whole PIT histogram, not one level.
    * Resolution is informative only when the forecast probability varies
      across cells -- e.g. the empirical predictive probability of the realised
      bin, or a per-cell level from a conformal wrapper.

    pit_or_probs: forecast probability of the event, any shape, values in
    [0, 1]. outcomes: binary (bool or 0/1), same shape. Both are flattened and
    pairs with a non-finite entry are dropped. n_bins: K equal-width bins
    [k/K, (k+1)/K) (last bin closed at 1), or None to bin on the distinct
    forecast values -- Murphy's original form, which makes the three-term
    identity exact.

    Returns dict: reliability, resolution, uncertainty, brier,
    within_bin_variance, within_bin_covariance, n.
    """
    p = np.asarray(pit_or_probs, dtype=float).ravel()
    o = np.asarray(outcomes, dtype=float).ravel()
    if p.shape != o.shape:
        raise ValueError(f"forecast/outcome size mismatch: {p.shape} vs {o.shape}")
    keep = np.isfinite(p) & np.isfinite(o)
    p, o = p[keep], o[keep]
    n = p.size
    if n == 0:
        raise ValueError("no finite forecast/outcome pairs")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("forecast probabilities must lie in [0, 1]")
    if not np.all((o == 0.0) | (o == 1.0)):
        raise ValueError("outcomes must be binary (bool or 0/1)")

    if n_bins is None:
        _, idx = np.unique(p, return_inverse=True)
        idx = idx.ravel()
        k = int(idx.max()) + 1
    else:
        idx = _bin_unit_interval(p, int(n_bins))
        k = int(n_bins)
    counts = np.bincount(idx, minlength=k).astype(float)
    occupied = counts > 0
    f_bar = np.zeros(k)
    o_bar = np.zeros(k)
    f_bar[occupied] = np.bincount(idx, weights=p, minlength=k)[occupied] / counts[occupied]
    o_bar[occupied] = np.bincount(idx, weights=o, minlength=k)[occupied] / counts[occupied]
    base = o.mean()

    reliability = float(np.sum(counts * (f_bar - o_bar) ** 2) / n)
    resolution = float(np.sum(counts * (o_bar - base) ** 2) / n)
    uncertainty = float(base * (1.0 - base))
    wbv = float(np.sum((p - f_bar[idx]) ** 2) / n)
    wbc = float(2.0 * np.sum((p - f_bar[idx]) * (o - o_bar[idx])) / n)
    brier = float(np.mean((p - o) ** 2))
    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier": brier,
        "within_bin_variance": wbv,
        "within_bin_covariance": wbc,
        "n": int(n),
    }


def murphy_pit(pit_values: np.ndarray, n_bins: int = 10) -> dict[str, float | np.ndarray]:
    """Reliability-resolution-uncertainty decomposition on the PIT scale with the
    UNIFORM distribution as reference.

    A calibrated forecaster's PIT values are Uniform(0, 1) (Gneiting,
    Balabdaoui & Raftery 2007, JRSS-B 69, 243-268), so on the PIT scale every
    model makes the SAME claim: the realised PIT lands in each of K equal-width
    bins with probability 1/K. Take that claim u = (1/K, ..., 1/K) as the
    forecast, the realised PIT bin as a K-category outcome, score with the
    multicategory Brier score, and apply the divergence form of Murphy's
    decomposition (Broecker 2009, QJRMS 135, 1512-1519; the same
    calibration-vs-reference structure Gneiting & Ranjan 2013, EJS 7,
    1747-1782, use for combined predictive distributions): for a proper score
    S with divergence d and entropy e,

        E S(F, Y) = E d(F, G_F)  -  E d(G_F, R)  +  e(R)
                    reliability    resolution      uncertainty

    where G_F is the realised outcome distribution given the forecast and R is
    the reference distribution. Here F = u, G_F = g = the PIT histogram, and
    THE REFERENCE IS THE UNIFORM, R = u -- the null of calibration, not the
    empirical histogram (the "climatology" reference of ``murphy_decomposition``,
    which on this scale would give resolution == 0 identically). With the
    quadratic divergence d(p, q) = sum_k (p_k - q_k)^2:

        reliability = sum_k (g_k - 1/K)^2         = chi2 / (N K); 0 iff flat
        uncertainty = e(u) = 1 - 1/K              Brier score of the uniform claim
        resolution  = e(u) - e(g) = sum_k g_k^2 - 1/K
                      gain a recalibrated claim (g in place of u) makes over u
        brier       = mean_i sum_k (1/K - 1{b_i = k})^2 = 1 - 1/K
        identity      brier = reliability - resolution + uncertainty (exact).

    Because the uniform claim scores identically whatever bin materialises,
    brier == uncertainty and hence reliability == resolution numerically under
    this reference: the claim's calibration shortfall equals the gain the
    recalibrated forecast would make. That is a property of the PIT scale (the
    transform standardises every forecast to u and discards sharpness), not a
    bug. The informative quantities are reliability -- the divergence from
    uniformity, which one level's coverage gap cannot see (hump = overdispersed,
    U = underdispersed, slope = biased; GBR 2007, Fig. 2) -- and the potential
    score uncertainty - resolution = e(g). A resolution term with independent
    content needs the predictive samples, not the PIT alone (Hersbach 2000's
    rank-histogram CRPS decomposition); that is deliberately out of scope.

    pit_values: any shape, values in [0, 1]; non-finite entries dropped.
    Returns dict: reliability, resolution, uncertainty, brier, hist (the K
    bin frequencies, sums to 1), n.
    """
    z = np.asarray(pit_values, dtype=float).ravel()
    z = z[np.isfinite(z)]
    if z.size == 0:
        raise ValueError("no finite PIT values")
    if np.any((z < 0.0) | (z > 1.0)):
        raise ValueError("PIT values must lie in [0, 1]")
    k = int(n_bins)
    idx = _bin_unit_interval(z, k)
    hist = np.bincount(idx, minlength=k) / z.size
    u = 1.0 / k
    onehot = np.eye(k)[idx]                                     # [n, K]
    brier = float(np.mean(np.sum((u - onehot) ** 2, axis=1)))
    reliability = float(np.sum((hist - u) ** 2))
    uncertainty = float(k * u * (1.0 - u))                      # e(u) = 1 - 1/K
    resolution = float(uncertainty - np.sum(hist * (1.0 - hist)))
    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier": brier,
        "hist": hist,
        "n": int(z.size),
    }
