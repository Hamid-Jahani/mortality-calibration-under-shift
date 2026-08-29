"""Post-hoc conformal UQ wrappers on the log m_x scale.

Three distribution-free wrappers around ANY study-interface model
(methodology rule 4: ``fit(D, E)`` / ``sample_mx(h, n, rng)``):

* :class:`SplitConformalMx` — split (inductive) conformal prediction
  (Vovk, Gammerman & Shafer 2005; Lei, G'Sell, Rinaldo, Tibshirani &
  Wasserman 2018, JASA), Mondrian-stratified by age band (Vovk et al. 2005,
  ch. 4) and by forecast horizon.
* :class:`EnbPIMx` — ensemble batch prediction intervals (Xu & Xie 2021,
  ICML, "Conformal prediction interval for dynamic time-series"), adapted
  to annual multi-step mortality forecasting; see the class docstring for
  the exact deviations from the online original.
* :class:`CopulaPathConformal` — simultaneous (joint-path) bands from a
  scaled sup-norm nonconformity score, the construction of Diquigiovanni,
  Fontana & Vantini (2022) for conformal functional bands, in the spirit of
  copula-based multi-target conformal (Messoudi, Destercke & Rousseau 2021)
  and multi-horizon time-series conformal (Stankeviciute, Alaa &
  van der Schaar 2021).

Split discipline (PREREGISTRATION.md, "Factors")
------------------------------------------------
Calibration data come ONLY from years at or before the training cutoff: each
wrapper receives the training panel and splits it internally (inner time
splits). No wrapper ever sees a test year. All internal splits respect time
order — the calibration years are always LATER than the years the underlying
base fits are trained on, mirroring the forecasting task.

What conformal gives — and what it does not
-------------------------------------------
Conformal calibration yields prediction INTERVALS with finite-sample validity
under exchangeability of nonconformity scores — not a full predictive
distribution. The study harness consumes samples (rule 4), so ``sample_mx``
draws UNIFORMLY inside the conformal interval of each (horizon, age) cell,
independently across cells and draws. This sampling convention:

* preserves the interval: the support of the draws is exactly [lo, hi]
  on the log m_x scale;
* is meaningful only for interval/coverage metrics at the construction
  level ``1 - alpha`` (empirical coverage, Winkler, and — for
  :class:`CopulaPathConformal` — joint path coverage of the simultaneous
  band);
* makes CRPS / log score / PIT NON-PRIMARY for conformal cells: the uniform
  density is a deliberately agnostic placeholder, not a distributional
  claim. Proper scores for conformal grid cells must be reported as a
  flagged secondary column, never ranked against distributional mechanisms.

Because the evaluation harness extracts central intervals from samples by
empirical quantiles, the interval it reads at level ``1 - alpha`` is the
conformal interval shrunk inward by ``alpha`` of its width (quantiles of a
uniform). Sample-based empirical coverage of these wrappers therefore
slightly UNDERSTATES the coverage of the conformal interval itself; the
finite-sample quantile inflation ``ceil((n+1)(1-alpha))/n`` works in the
opposite direction. Both effects are part of the documented finite-sample
slack in the wrapper tests.

Path semantics: for :class:`SplitConformalMx` and :class:`EnbPIMx` the bands
are pointwise (marginal per horizon); joint path coverage of those two is
reported but carries no guarantee. :class:`CopulaPathConformal` is the
mechanism whose band is calibrated to be simultaneous over horizons.
"""
from __future__ import annotations

import ctypes
import gc
import math
import sys

import numpy as np


def release_memory() -> None:
    """Return freed heap pages to the OS between large fits (Linux only).

    Measured 2026-08-29 (SWE males, GP with the 60-year cap): one exact-GP
    fit peaks at ~2.5 GB RSS and the allocator keeps it after del+gc, so a
    ten-member conformal wrapper climbs to ~8 GB and two workers swap a
    23 GB node. glibc's malloc_trim(0) hands the freed arenas back; it
    changes no number. No-op where libc is unavailable (Windows).
    """
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:  # pragma: no cover - no glibc
            pass

#: Mondrian age bands (inclusive edges). The last band is open-ended above,
#: so panels with more ages than the last edge still get a band.
DEFAULT_AGE_BANDS = ((0, 24), (25, 64), (65, 99))

_RATE_FLOOR = 1e-10  # same floor as mortcal.models.lc


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def _log_rates(D: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Observed log rates on the half-count scale (addendum 2 §2 / 3 §6).

    log(max(D, 0.5)/E) where E > 0, NaN where E == 0 (structural zeros — no
    observation exists; _conformal_quantile ignores NaN residuals). The old
    1e-10 floor put a single zero-death calibration cell at log-rate -23.03,
    inflating a SWE-female band radius from 1.44 to 12.94 nats.
    """
    D = np.asarray(D, dtype=float)
    E = np.asarray(E, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(E > 0, np.maximum(D, 0.5) / np.where(E > 0, E, 1.0), np.nan)
    return np.log(r)


def _band_indices(n_ages: int, bands) -> list[np.ndarray]:
    """Index arrays of ages per Mondrian band; validates the partition.

    Bands are (lo, hi) inclusive, must start at age 0, be contiguous and
    non-overlapping. The last band absorbs any ages above its upper edge
    (open-ended top, mirroring the treatment of open age groups).
    """
    bands = tuple(tuple(b) for b in bands)
    if bands[0][0] != 0:
        raise ValueError("first age band must start at 0")
    for (lo, hi), (lo2, _hi2) in zip(bands, bands[1:]):
        if hi + 1 != lo2:
            raise ValueError(f"age bands must be contiguous, got ...{hi}], [{lo2}...")
    out = []
    for i, (lo, hi) in enumerate(bands):
        hi_eff = n_ages - 1 if i == len(bands) - 1 else min(hi, n_ages - 1)
        idx = np.arange(lo, hi_eff + 1)
        idx = idx[idx < n_ages]
        if idx.size == 0 and lo < n_ages:
            raise ValueError("empty age band after clipping")
        out.append(idx)
    return [b for b in out if b.size > 0]


def _conformal_quantile(res: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile of nonconformity scores.

    The k-th order statistic with ``k = ceil((n + 1) * (1 - alpha))``
    (Vovk et al. 2005; Lei et al. 2018, eq. 2.2) — the empirical quantile at
    level ``ceil((n+1)(1-alpha))/n``. When ``k > n`` the exact level is
    unattainable with n scores; we clamp to the maximum score. That clamp is
    anti-conservative in principle and is documented here: with the study's
    band sizes (>= 25 scores per cell) it binds only for alpha < 1/(n+1).
    """
    r = np.asarray(res, dtype=float).ravel()
    r = np.sort(r[np.isfinite(r)])          # NaN = structural zero, no residual
    n = r.size
    if n == 0:
        # A band the base family is undefined on (CBD is fit on ages 55+, so
        # the 0-24 band has no residual at all). NaN radius -> NaN samples ->
        # the runner's age mask drops exactly those ages, which is the same
        # treatment the family's native cell gets. Raising here instead would
        # kill every CBD x conformal cell in the grid over ages the cell was
        # never going to score.
        return float("nan")
    k = math.ceil((n + 1) * (1.0 - alpha))
    return float(r[min(k, n) - 1])


def _nanmedian_log(x: np.ndarray) -> np.ndarray:
    """Median over the sample axis, NaN-safe (undefined ages stay NaN)."""
    with np.errstate(invalid="ignore"):
        return np.nanmedian(x, axis=0)


def _median_log_forecast(model, h: int, n_samples: int,
                         rng: np.random.Generator) -> np.ndarray:
    """[h, n_ages] pointwise median of log m_x predictive samples.

    The median commutes with the log, so this is also the log of the median
    m_x forecast. The median is invariant to the (mis)scaling of the base
    model's uncertainty mechanism — conformal calibrates around the point
    forecast and replaces the native interval entirely.
    """
    # Families whose predictive law has a closed-form median (the GP: its
    # posterior mean) expose median_logmx(h); using it is exact and avoids the
    # n_samples joint draws that OOMed the GP conformal cells (1.59 GB, solo).
    if hasattr(model, "median_logmx"):
        return np.asarray(model.median_logmx(h), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return _nanmedian_log(np.log(model.sample_mx(h, n_samples, rng)))


def _cached_center(wrapper, h: int) -> np.ndarray:
    """Median log forecast of the wrapper's full-panel refit, cached per h.

    Uses a deterministic internal generator (seed + 1) so the interval
    CENTER does not depend on the rng passed to ``sample_mx``.
    """
    cached = wrapper._centers.get(h)
    if cached is None:
        rng = np.random.default_rng(wrapper.seed + 1)
        cached = _median_log_forecast(wrapper.model, h, wrapper.n_median_samples, rng)
        wrapper._centers[h] = cached
    return cached


def _expand_radius(q: np.ndarray, h: int) -> np.ndarray:
    """[h, n_ages] radii from calibrated radii ``q`` [h_cal, n_ages].

    Horizons beyond the calibrated range reuse the LAST calibrated radius.
    That is anti-conservative for a growing forecast-error variance and is a
    documented limitation; in the registered design (test horizons h <= 9,
    calibration horizons 9 per addendum 3 §9) it never binds.
    """
    idx = np.minimum(np.arange(h), q.shape[0] - 1)
    return q[idx]


def _uniform_interval_samples(center: np.ndarray, radius: np.ndarray,
                              n: int, rng: np.random.Generator) -> np.ndarray:
    """[n, h, n_ages] m_x draws uniform on [center - r, center + r] in log space."""
    u = rng.uniform(-1.0, 1.0, size=(n,) + center.shape)
    return np.exp(center[None, :, :] + u * radius[None, :, :])


def _interval_bounds(wrapper, h: int) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) log m_x bounds [h, n_ages] at the construction level 1 - alpha.

    Addendum 3 §6: conformal cells are SCORED from these bounds directly —
    no uniform-in-interval sampling (whose empirical quantiles shrink the
    interval) and no Poisson composition (the radius is calibrated on
    observed residuals and already contains observation noise).
    """
    center = _cached_center(wrapper, h)
    radius = _expand_radius(wrapper._q, h)
    return center - radius, center + radius


def _trailing_block_residuals(base_factory, D: np.ndarray, E: np.ndarray,
                              K: int, h_cal: int, n_samples: int,
                              rng: np.random.Generator) -> np.ndarray:
    """Leave-trailing-block-out residuals: [K, h_cal, n_ages] absolute errors.

    Member i is fit on years [0, t_i) with the trailing block [t_i, T) left
    out, for K staggered origins t_i = T - h_cal - K + 1 + i. Each member's
    median log forecast at horizons 1..h_cal is scored against the OBSERVED
    log rates of the h_cal years following its origin — all of which lie
    inside the training panel, so no test year is touched. Overlapping
    blocks mean residuals are dependent across members; EnbPI's validity
    argument tolerates this (Xu & Xie 2021, Thm 1 allows estimation error
    and dependence with vanishing bounds).
    """
    n_ages, T = D.shape
    first = T - h_cal - K + 1
    if first < 10:
        raise ValueError(
            f"panel too short: earliest member would train on {first} years "
            f"(need >= 10); reduce K={K} or h_cal={h_cal}")
    res = np.empty((K, h_cal, n_ages))
    obs = _log_rates(D, E)  # [ages, T]
    for i in range(K):
        t = first + i
        member = base_factory().fit(D[:, :t], E[:, :t])
        med = _median_log_forecast(member, h_cal, n_samples, rng)  # [h_cal, ages]
        res[i] = np.abs(obs[:, t:t + h_cal].T - med)
        del member
        release_memory()            # bounded footprint across the K member fits
    return res


# --------------------------------------------------------------------------
# 1. split conformal
# --------------------------------------------------------------------------

class SplitConformalMx:
    """Split conformal intervals for log m_x, Mondrian by horizon x age band.

    Lei et al. (2018) split conformal adapted to panel forecasting:

    1. Hold out the LAST ``cal_years`` (C=8) of the training panel.
    2. Fit a fresh base model (from ``base_factory``) on the remaining years.
    3. For each horizon h=1..C, score the model's MEDIAN log m_x forecast
       against the observed log rates of calibration year h; absolute
       residuals are pooled within each Mondrian age band (default 0-24,
       25-64, 65-99) separately per horizon.
    4. The interval radius per (h, band) is the residual quantile at level
       ``ceil((n_cal + 1)(1 - alpha)) / n_cal`` (finite-sample conformal
       quantile), where n_cal = number of ages in the band.
    5. The interval CENTER is the median forecast of the base model REFIT on
       the full training panel — the standard practical split-conformal
       refit; the horizon-h radius from the calibration origin is applied to
       the horizon-h test forecast (horizon-matched, as in Stankeviciute
       et al. 2021).

    Caveat (documented, not hidden): with a single calibration origin the
    horizon-h residuals across ages share ONE realization of the latent
    period-effect path, so exchangeability across ages within a band is
    approximate; the idiosyncratic (Poisson) component of observed rates and
    the finite-sample quantile inflation are what make the intervals hold up
    empirically. :class:`EnbPIMx` addresses this with multi-origin residuals.

    Parameters
    ----------
    base_factory : callable () -> study-interface model (a class is fine)
    alpha : miscoverage level of the constructed interval (one interval per
        level; re-fit the wrapper to change level)
    cal_years : C, number of trailing training years held out for calibration
    bands : Mondrian age bands, inclusive (lo, hi) edges
    n_median_samples : predictive samples used to estimate median forecasts
    seed : internal seed for median estimation (kept separate from the rng
        passed to ``sample_mx`` so the interval is deterministic given fit)
    """

    def __init__(self, base_factory, alpha: float = 0.05, cal_years: int = 9,
                 bands=DEFAULT_AGE_BANDS, n_median_samples: int = 1000,
                 seed: int = 20260825):
        self.base_factory = base_factory
        self.alpha = alpha
        self.cal_years = cal_years
        self.bands = bands
        self.n_median_samples = n_median_samples
        self.seed = seed

    def fit(self, D: np.ndarray, E: np.ndarray) -> "SplitConformalMx":
        n_ages, T = D.shape
        C = self.cal_years
        if T - C < 10:
            raise ValueError(f"panel too short: {T - C} proper-training years (need >= 10)")
        band_idx = _band_indices(n_ages, self.bands)
        rng = np.random.default_rng(self.seed)

        proper = self.base_factory().fit(D[:, :T - C], E[:, :T - C])
        med = _median_log_forecast(proper, C, self.n_median_samples, rng)  # [C, ages]
        obs = _log_rates(D[:, T - C:], E[:, T - C:]).T                     # [C, ages]
        res = np.abs(obs - med)

        q = np.empty((C, n_ages))
        for h in range(C):
            for idx in band_idx:
                q[h, idx] = _conformal_quantile(res[h, idx], self.alpha)
        self._q = q

        self.model = self.base_factory().fit(D, E)  # full-panel refit: the center
        self._centers: dict[int, np.ndarray] = {}
        return self

    def sample_mx(self, h: int, n: int, rng: np.random.Generator) -> np.ndarray:
        """[n, h, n_ages] draws uniform inside the conformal interval per cell.

        See the module docstring: intervals, not a distribution — use only
        for interval/coverage metrics at the construction level.
        """
        center = _cached_center(self, h)
        radius = _expand_radius(self._q, h)
        return _uniform_interval_samples(center, radius, n, rng)

    def interval(self, h: int) -> tuple[np.ndarray, np.ndarray]:
        """(lo, hi) log m_x bounds [h, n_ages] at level 1 - alpha (addendum 3 §6)."""
        return _interval_bounds(self, h)


# --------------------------------------------------------------------------
# 2. EnbPI (adapted)
# --------------------------------------------------------------------------

class EnbPIMx:
    """EnbPI-style intervals from leave-block-out ensemble residuals.

    Xu & Xie (2021) construct intervals from out-of-bag residuals of an
    ensemble of models each trained with a block of time left out. Adapted
    here to annual multi-step mortality forecasting:

    * Members: K=10 fits on overlapping year-blocks — member i trains on
      years [0, t_i) and leaves out the trailing block [t_i, T), origins t_i
      staggered one year apart so blocks overlap. (True interior
      leave-block-out would put holes in the panel, which random-walk-based
      mortality forecasters cannot fit; trailing blocks preserve both
      contiguity and time order. Documented deviation #1.)
    * Residuals: each member's median log m_x forecast at horizons 1..h_cal
      is scored on the observed years after its own origin — K residuals per
      (horizon, age), aggregated per (horizon, Mondrian age band); n_cal =
      K x band size. Horizon-matched rather than EnbPI's single-step
      residual pool (deviation #2 — multi-step-ahead needs per-horizon
      error distributions).
    * Interval: same finite-sample conformal quantile construction as
      :class:`SplitConformalMx`, centred on the median forecast of a
      full-panel refit (φ-aggregation replaced by one refit — the refit is
      horizon-matched to the residuals, whereas the stale members are not;
      deviation #3).
    * NO feedback/online residual updates: the original EnbPI re-ingests
      residuals as test observations arrive; with annual data and the
      pre-registered one-shot evaluation of the test window that channel
      does not exist here (deviation #4).

    The multi-origin residuals are what split conformal lacks: K distinct
    realizations of the latent period-effect path per horizon, so the
    residual pool reflects path uncertainty, not one draw of it.
    """

    def __init__(self, base_factory, alpha: float = 0.05, K: int = 10,
                 h_cal: int = 9, bands=DEFAULT_AGE_BANDS,
                 n_median_samples: int = 1000, seed: int = 20260825):
        self.base_factory = base_factory
        self.alpha = alpha
        self.K = K
        self.h_cal = h_cal
        self.bands = bands
        self.n_median_samples = n_median_samples
        self.seed = seed

    def fit(self, D: np.ndarray, E: np.ndarray) -> "EnbPIMx":
        n_ages, _T = D.shape
        band_idx = _band_indices(n_ages, self.bands)
        rng = np.random.default_rng(self.seed)
        res = _trailing_block_residuals(self.base_factory, D, E, self.K,
                                        self.h_cal, self.n_median_samples, rng)
        q = np.empty((self.h_cal, n_ages))
        for h in range(self.h_cal):
            for idx in band_idx:
                q[h, idx] = _conformal_quantile(res[:, h, idx], self.alpha)
        self._q = q
        self.model = self.base_factory().fit(D, E)
        self._centers: dict[int, np.ndarray] = {}
        return self

    def sample_mx(self, h: int, n: int, rng: np.random.Generator) -> np.ndarray:
        """[n, h, n_ages] draws uniform inside the EnbPI interval per cell."""
        center = _cached_center(self, h)
        radius = _expand_radius(self._q, h)
        return _uniform_interval_samples(center, radius, n, rng)

    def interval(self, h: int) -> tuple[np.ndarray, np.ndarray]:
        """(lo, hi) log m_x bounds [h, n_ages] at level 1 - alpha (addendum 3 §6)."""
        return _interval_bounds(self, h)


# --------------------------------------------------------------------------
# 3. copula / joint-path conformal
# --------------------------------------------------------------------------

class CopulaPathConformal:
    """Simultaneous forecast bands calibrated for JOINT path coverage.

    Pointwise intervals (everything above) guarantee marginal coverage per
    horizon; the probability that an entire h=1..H trajectory stays inside
    them is far lower (hypothesis H3). This wrapper calibrates the band to
    the PATH:

    * Calibration paths: leave-trailing-block-out residuals as in
      :class:`EnbPIMx` — K origins x n_ages ages, each contributing one
      calibration path (its h=1..h_cal absolute-residual trajectory). Multi-
      origin matters even more here than marginally: a path score needs many
      realizations of the latent period-effect path to have an exchangeable
      calibration distribution.
    * Nonconformity: scaled sup-norm  S = max_h |resid_h| / s_h  with s_h
      the per-(horizon, age-band) median absolute residual — the modulation
      function of Diquigiovanni et al. (2022), which shapes the band like
      the error process so no single horizon dominates the max. This scores
      the whole path with one number, implicitly bounding the residual
      copula's sup-norm (Messoudi et al. 2021).
    * Quantile: per Mondrian band, the finite-sample conformal quantile
      q_band of the K x band_size path scores.
    * Band: center_h ± q_band * s_h simultaneously over h. For a new path
      exchangeable with the calibration paths, P(path entirely inside the
      band) >= 1 - alpha by the standard conformal argument applied to the
      scalar score S — this is a guarantee on JOINT path coverage, at the
      price of wider-than-pointwise intervals at every horizon.

    Center: median forecast of a full-panel refit, as in the other wrappers.
    Sampling convention: uniform per cell inside the band (module
    docstring); joint path coverage evaluated from these samples inherits
    the quantile-shrinkage caveat, so measured joint coverage slightly
    understates the band's.
    """

    def __init__(self, base_factory, alpha: float = 0.05, K: int = 10,
                 h_cal: int = 9, bands=DEFAULT_AGE_BANDS,
                 n_median_samples: int = 1000, seed: int = 20260825):
        self.base_factory = base_factory
        self.alpha = alpha
        self.K = K
        self.h_cal = h_cal
        self.bands = bands
        self.n_median_samples = n_median_samples
        self.seed = seed

    def fit(self, D: np.ndarray, E: np.ndarray) -> "CopulaPathConformal":
        n_ages, _T = D.shape
        band_idx = _band_indices(n_ages, self.bands)
        rng = np.random.default_rng(self.seed)
        res = _trailing_block_residuals(self.base_factory, D, E, self.K,
                                        self.h_cal, self.n_median_samples, rng)

        s = np.empty((self.h_cal, n_ages))          # per-(h, band) residual scale
        for h in range(self.h_cal):
            for idx in band_idx:
                s[h, idx] = max(float(np.nanmedian(res[:, h, idx])), 1e-12)  # nan-aware: CBD ages < age_min are NaN

        with np.errstate(invalid="ignore"):
            scores = np.nanmax(res / s[None, :, :], axis=1)  # [K, ages] sup-norm per path; NaN only where ALL horizons undefined
        radius = np.empty((self.h_cal, n_ages))
        for idx in band_idx:
            q_band = _conformal_quantile(scores[:, idx], self.alpha)
            radius[:, idx] = q_band * s[:, idx]
        self._q = radius

        self.model = self.base_factory().fit(D, E)
        self._centers: dict[int, np.ndarray] = {}
        return self

    def sample_mx(self, h: int, n: int, rng: np.random.Generator) -> np.ndarray:
        """[n, h, n_ages] draws uniform inside the simultaneous band per cell."""
        center = _cached_center(self, h)
        radius = _expand_radius(self._q, h)
        return _uniform_interval_samples(center, radius, n, rng)

    def interval(self, h: int) -> tuple[np.ndarray, np.ndarray]:
        """(lo, hi) log m_x bounds [h, n_ages] at level 1 - alpha (addendum 3 §6)."""
        return _interval_bounds(self, h)
