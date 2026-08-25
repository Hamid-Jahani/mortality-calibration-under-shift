"""Cairns-Blake-Dowd two-factor model M5 (Cairns, Blake & Dowd 2006, JRI).

    logit q(x, t) = k1_t + k2_t * (x - xbar),    xbar = mean of the fitted ages

Exposes the single study-wide interface (methodology rule 4):

    model.fit(D, E)                  # [n_ages, n_years] death counts & exposures
    model.sample_mx(h, n, rng)       # -> [n, h, n_ages] predictive m_x samples
    model.fitted_mx()                # -> [n_ages, n_years] in-sample fitted m_x

Estimation is the calibration of Cairns, Blake & Dowd (2006, "A Two-Factor
Model for Stochastic Mortality with Parameter Uncertainty: Theory and
Calibration", Journal of Risk and Insurance 73(4), 687-718): crude one-year
death probabilities q_hat = 1 - exp(-m_hat) from m_hat = D/E (constant force
of mortality within the year), then, for each year t, OLS of logit q_hat on
centred age. No identifiability constraint is needed: (k1_t, k2_t) are the
intercept-at-xbar and slope of that regression, directly identified.

Predictive uncertainty is the MODEL-NATIVE mechanism of the paper: a BIVARIATE
random walk with drift on (k1, k2), with both the 2x2 innovation covariance
and the drift-estimation uncertainty (Sigma / n_increments) included — the
"parameter uncertainty" of the paper's title. Paths are simulated jointly over
horizons so joint (path) coverage is well-defined. Bootstrap / ensemble /
conformal wrappers live elsewhere so that model family x UQ mechanism stay
crossed factors, never conflated.

Age restriction. CBD assumes logit q is near-linear in age, which holds at
higher ages only (typical use: ages 55-99). Two ways to restrict the fit:

* ``CBD(age_min=55).fit(D, E)`` — the runner's path, which keeps the
  study-wide ``fit(D, E)`` signature. The panel handed to ``fit`` (HMD rows
  for ages 0-99; ``age0`` is the true age of row 0) REMAINS the model's age
  dimension: ``sample_mx`` and ``fitted_mx`` return every panel row. CBD is
  undefined below ``age_min`` by design, so ``sample_mx`` fills those rows
  with NaN and the runner masks them (``fit_mask`` marks the fitted rows).
  ``fitted_mx`` cannot carry NaN: the Poisson bootstrap wrapper resamples
  D_b ~ Poisson(E * fitted_mx) and ``numpy.random.Generator.poisson``
  rejects NaN means. Its default therefore fills the excluded rows with the
  OBSERVED crude ratio D/E, so a pseudo-panel reproduces the data outside
  the model's range — rows the refit ignores anyway.
  ``fitted_mx(excluded="nan")`` returns the strict model surface.
* ``fit(D, E, ages=..., age0=...)`` — explicit sub-panel selection: the
  selected ages ARE the model's age dimension and every output has
  ``len(ages)`` rows, no NaN. Typical HMD use with rows for ages 0-99:
  ``fit(D, E, ages=np.arange(55, 100))``.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-10


class CBD:
    """Two-factor CBD (M5): per-year OLS calibration + bivariate RWD forecast.

    Parameters
    ----------
    age_min : int or None
        Lowest TRUE age to fit when ``fit(D, E)`` receives a wider panel
        (rows below it are excluded from estimation but kept in the output
        age dimension — see the module docstring). None (default) fits every
        row. The runner passes 55.
    """

    def __init__(self, age_min: int | None = None):
        self.age_min = None if age_min is None else int(age_min)

    def fit(self, D: np.ndarray, E: np.ndarray, ages: np.ndarray | None = None,
            age0: int = 0) -> "CBD":
        """Calibrate on deaths D and central exposures E, both [n_ages, n_years].

        ages : ages to fit, mapped to rows of D via ``age - age0`` where age0
               is the true age of row 0. Default: every row of D at or above
               ``age_min`` (all rows when ``age_min`` is None). ``ages`` and
               ``age_min`` are mutually exclusive — they define the output age
               dimension differently (module docstring).
        """
        D = np.asarray(D, dtype=float)
        E = np.asarray(E, dtype=float)
        n_rows, n_years = D.shape
        if n_years < 3:
            raise ValueError("need >= 3 years: drift + innovation cov use ddof=1")
        if ages is not None and self.age_min is not None:
            raise ValueError("choose one age restriction: age_min (constructor) "
                             "or ages (fit), not both")
        if ages is None:
            first = 0 if self.age_min is None else self.age_min - int(age0)
            if not 0 <= first < n_rows:
                raise ValueError("age_min falls outside the rows of D for this age0")
            rows = np.arange(first, n_rows)
            ages = age0 + rows
            self.n_out = n_rows                      # output = the whole panel
            self._out_pos = rows                     # fitted rows keep their panel position
            self._mx_crude = self._crude_ratio(D, E)
        else:
            ages = np.asarray(ages)
            rows = ages.astype(int) - int(age0)
            if rows.min() < 0 or rows.max() >= n_rows:
                raise ValueError("requested ages fall outside the rows of D for this age0")
            self.n_out = len(rows)                   # output = the selected ages
            self._out_pos = np.arange(len(rows))
            self._mx_crude = self._crude_ratio(D[rows], E[rows])
        self.fit_mask = np.zeros(self.n_out, dtype=bool)
        self.fit_mask[self._out_pos] = True
        D, E = D[rows], E[rows]

        m_hat = np.clip(D / E, _EPS, None)
        q_hat = np.clip(1.0 - np.exp(-m_hat), _EPS, 1.0 - _EPS)
        y = np.log(q_hat) - np.log1p(-q_hat)                    # logit q, [ages, years]

        self.ages = ages.astype(float)
        self.xbar = float(self.ages.mean())
        xc = self.ages - self.xbar
        # OLS per year against centred age: intercept = column mean (sum xc = 0)
        self.k1 = y.mean(axis=0)                                 # [years]
        self.k2 = (xc[:, None] * y).sum(axis=0) / (xc ** 2).sum()

        K = np.column_stack([self.k1, self.k2])                  # [T, 2]
        Z = np.diff(K, axis=0)                                   # [T-1, 2] increments
        self.K_last = K[-1]
        self.mu = Z.mean(axis=0)                                 # drift estimate
        self.n_inc = Z.shape[0]
        self.Sigma = np.cov(Z, rowvar=False, ddof=1)             # 2x2 innovation cov
        self._chol = self._safe_cholesky(self.Sigma)
        return self

    @staticmethod
    def _crude_ratio(D: np.ndarray, E: np.ndarray) -> np.ndarray:
        """Observed D/E with 0 where E == 0 (any finite value works there: the
        bootstrap's Poisson mean is E * m = 0 regardless)."""
        return np.divide(D, E, out=np.zeros_like(D), where=E > 0)

    @staticmethod
    def _safe_cholesky(S: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.cholesky(S)
        except np.linalg.LinAlgError:                            # degenerate increments
            jitter = 1e-12 * max(float(np.trace(S)), 1.0)
            return np.linalg.cholesky(S + jitter * np.eye(2))

    def sample_mx(self, h: int, n: int, rng: np.random.Generator) -> np.ndarray:
        """[n, h, n_ages] pathwise m_x samples with drift + innovation uncertainty.

        Per path: mu ~ N(mu_hat, Sigma / n_inc) once (epistemic, drift
        estimation), innovations ~ N(0, Sigma) each step (aleatoric); K
        accumulated pathwise. Back-transform is exact and overflow-safe:
        q = expit(z)  =>  m_x = -log(1 - q) = log(1 + e^z) = softplus(z).
        Rows excluded via ``age_min`` are NaN (CBD undefined there; the
        runner masks them with ``fit_mask``).
        """
        L = self._chol
        mu = self.mu + rng.standard_normal((n, 2)) @ L.T / np.sqrt(self.n_inc)
        eps = rng.standard_normal((n, h, 2)) @ L.T
        K = self.K_last + np.cumsum(mu[:, None, :] + eps, axis=1)   # [n, h, 2]
        xc = self.ages - self.xbar
        z = K[:, :, :1] + K[:, :, 1:] * xc[None, None, :]           # logit q
        out = np.full((n, h, self.n_out), np.nan)
        out[:, :, self._out_pos] = np.logaddexp(0.0, z)             # -log(1-q)
        return out

    def fitted_mx(self, excluded: str = "observed") -> np.ndarray:
        """In-sample fitted m_x surface, [n_ages, n_years].

        Fitted rows: the same exact back-transform as ``sample_mx`` applied to
        the fitted factors, m_x = softplus(k1_t + k2_t (x - xbar)). Rows
        excluded via ``age_min``: ``excluded="observed"`` (default, what the
        Poisson bootstrap resamples from) returns the observed crude ratio
        D/E; ``excluded="nan"`` returns NaN — the strict model surface.
        """
        if excluded not in ("observed", "nan"):
            raise ValueError(f"excluded must be 'observed' or 'nan', got {excluded!r}")
        z = self.k1[None, :] + self.k2[None, :] * (self.ages - self.xbar)[:, None]
        out = (self._mx_crude.copy() if excluded == "observed"
               else np.full(self._mx_crude.shape, np.nan))
        out[self._out_pos] = np.logaddexp(0.0, z)
        return out
