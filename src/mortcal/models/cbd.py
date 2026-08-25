"""Cairns-Blake-Dowd two-factor model M5 (Cairns, Blake & Dowd 2006, JRI).

    logit q(x, t) = k1_t + k2_t * (x - xbar),    xbar = mean of the fitted ages

Exposes the single study-wide interface (methodology rule 4):

    model.fit(D, E)                  # [n_ages, n_years] death counts & exposures
    model.sample_mx(h, n, rng)       # -> [n, h, n_ages] predictive m_x samples

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

CBD assumes logit q is near-linear in age, which holds at higher ages only
(typical use: ages 55-99); ``fit(..., ages=..., age0=...)`` selects those rows.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-10


class CBD:
    """Two-factor CBD (M5): per-year OLS calibration + bivariate RWD forecast."""

    def fit(self, D: np.ndarray, E: np.ndarray, ages: np.ndarray | None = None,
            age0: int = 0) -> "CBD":
        """Calibrate on deaths D and central exposures E, both [n_ages, n_years].

        ages : ages to fit, mapped to rows of D via ``age - age0`` where age0
               is the true age of row 0. Default: every row of D. Typical HMD
               use with rows for ages 0-99: ``fit(D, E, ages=np.arange(55, 100))``.
        """
        D = np.asarray(D, dtype=float)
        E = np.asarray(E, dtype=float)
        n_rows, n_years = D.shape
        if n_years < 3:
            raise ValueError("need >= 3 years: drift + innovation cov use ddof=1")
        if ages is None:
            ages = age0 + np.arange(n_rows)
        ages = np.asarray(ages)
        rows = ages.astype(int) - int(age0)
        if rows.min() < 0 or rows.max() >= n_rows:
            raise ValueError("requested ages fall outside the rows of D for this age0")
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
        """
        L = self._chol
        mu = self.mu + rng.standard_normal((n, 2)) @ L.T / np.sqrt(self.n_inc)
        eps = rng.standard_normal((n, h, 2)) @ L.T
        K = self.K_last + np.cumsum(mu[:, None, :] + eps, axis=1)   # [n, h, 2]
        xc = self.ages - self.xbar
        z = K[:, :, :1] + K[:, :, 1:] * xc[None, None, :]           # logit q
        return np.logaddexp(0.0, z)                                 # -log(1-q)
