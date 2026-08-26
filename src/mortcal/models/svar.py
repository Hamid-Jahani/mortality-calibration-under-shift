"""Sparse (banded) VAR(1) on mortality improvements, after Li & Lu (2017).

Li, H. & Lu, Y. (2017), "Coherent Forecasting of Mortality Rates: A Sparse
Vector-Autoregression Approach", ASTIN Bulletin 47(2), 563-600, model age-
specific mortality improvements y_t = Delta log m_t with a VAR whose
coefficient matrix is sparse and concentrated near the diagonal, estimated by
penalised (elastic-net) regression. This implementation keeps their model
object — a VAR(1) on improvements with age-local dependence — but replaces the
penalised estimator with a hard BANDED restriction: age x's equation regresses
on its own lag and the lags of ages within a bandwidth W (default 3). The band
IS the sparsity structure; no penalisation dependency is needed. This is a
documented simplification of Li-Lu, not a reproduction of their estimator.

Study-wide interface (methodology rule 4):

    model.fit(D, E)                  # [n_ages, n_years] death counts & exposures
    model.sample_mx(h, n, rng)       # -> [n, h, n_ages] predictive m_x samples
    model.fitted_mx()                # -> [n_ages, n_years] one-step-ahead fitted m_x

Predictive uncertainty (MODEL-NATIVE mechanism) is simulated pathwise so joint
path coverage is well-defined, and includes BOTH
  * parameter uncertainty — per equation, coefficients are drawn once per path
    from N(beta_hat, sigma2_hat * (X'X)^{-1}), the standard OLS coefficient
    covariance; and
  * innovation noise — at every horizon a fresh joint innovation is drawn from
    N(0, Sigma_hat), where Sigma_hat is the cross-age residual covariance
    estimated on training residuals and shrunk toward its diagonal by a fixed
    Ledoit-Wolf-style convex combination
        Sigma_hat = (1 - w) * S + w * diag(S),   w = 0.2 (fixed, documented).
    Shrinking toward the diagonal leaves marginal variances untouched but
    regularises the cross-age correlations, which are estimated from few
    effective observations relative to the number of ages (Ledoit & Wolf 2004
    motivate the convex-combination form; the weight here is fixed, not
    optimised, to keep the estimator pre-registered and tuning-free).

Forecast log m paths are reconstructed by cumulative summation of simulated
improvements from the last observed log m (the improvements VAR forecasts
Delta log m; levels follow by integration).
"""
from __future__ import annotations

import numpy as np

_JITTER = 1e-12  # numerical ridge added before Cholesky factorisations


class SparseVAR:
    """Banded VAR(1) on y_t = Delta log m_t, per-age OLS with intercept.

    Parameters
    ----------
    W : int
        Bandwidth. Age x's equation uses lagged improvements of ages
        x-W .. x+W (clipped at the age range boundaries). W = 3 default.
    shrink : float
        Fixed weight of the diagonal target in the residual-covariance
        shrinkage (see module docstring). 0.2 default, pre-registered.
    """

    def __init__(self, W: int = 3, shrink: float = 0.2):
        self.W, self.shrink = W, shrink

    # ------------------------------------------------------------------ fit
    def fit(self, D: np.ndarray, E: np.ndarray) -> "SparseVAR":
        W = self.W
        # Half-count for zero-death cells (addendum 2 §2); structural E = 0
        # cells are missing improvements handled by the common-complete
        # subsample below (addendum 3 §1) — no PSD projection, no imputation.
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(E > 0, np.maximum(D, 0.5) / np.where(E > 0, E, 1.0), np.nan)
        logm = np.log(rate)                              # [ages, years], NaN at E=0
        n_age, n_year = logm.shape
        if n_year - 2 <= 2 * W + 2:
            raise ValueError(
                f"need > {2 * W + 4} years for banded VAR with W={W}; got {n_year}"
            )
        complete = np.isfinite(logm).all(axis=0)         # complete year-columns
        if not (complete[-1] and complete[-2]):
            raise ValueError("the last two training years must be complete "
                             "(they set the VAR state at the origin)")
        Y = np.diff(logm, axis=1)                        # improvements [ages, T-1]
        # regression row t uses Y[:, t-1] and Y[:, t] -> needs year columns
        # t-1, t, t+1 all complete (common-complete subsample, addendum 3 §1)
        row_ok = complete[:-2] & complete[1:-1] & complete[2:]     # [T-2]
        Ylag, Ynow = Y[:, :-1][:, row_ok], Y[:, 1:][:, row_ok]
        nobs = Ynow.shape[1]                             # retained regression rows
        if nobs <= 2 * W + 2:
            raise ValueError(
                f"only {nobs} complete regression rows for banded VAR with W={W}")

        self.n_age = n_age
        self._logm = logm.copy()                         # observed panel, for fitted_mx
        self.logm_last = logm[:, -1].copy()              # integration constant
        self.y_last = Y[:, -1].copy()                    # VAR state at the origin
        self.c_ = np.zeros(n_age)                        # intercepts (point)
        self.A_ = np.zeros((n_age, n_age))               # banded matrix (point)
        self._bands = []                                 # (lo, hi) per age
        self._beta = []                                  # OLS point estimates
        self._beta_chol = []                             # chol of coef covariance
        resid = np.empty((n_age, nobs))
        dof = np.empty(n_age)

        for i in range(n_age):
            lo, hi = max(0, i - W), min(n_age - 1, i + W)
            X = np.concatenate([np.ones((1, nobs)), Ylag[lo:hi + 1]]).T  # [nobs, p]
            p = X.shape[1]
            XtX = X.T @ X
            beta = np.linalg.solve(XtX, X.T @ Ynow[i])
            r = Ynow[i] - X @ beta
            sigma2 = float(r @ r) / (nobs - p)           # unbiased equation variance
            cov = sigma2 * np.linalg.inv(XtX)            # standard OLS coef covariance
            self._bands.append((lo, hi))
            self._beta.append(beta)
            self._beta_chol.append(np.linalg.cholesky(cov + _JITTER * np.eye(p)))
            self.c_[i] = beta[0]
            self.A_[i, lo:hi + 1] = beta[1:]
            resid[i] = r
            dof[i] = nobs - p

        # Cross-age residual covariance with per-equation dof correction
        # (congruence by a positive diagonal, so S stays PSD), then fixed
        # convex shrinkage toward the diagonal (see module docstring).
        scale = 1.0 / np.sqrt(dof)
        S = (resid @ resid.T) * np.outer(scale, scale)
        w = self.shrink
        sigma_shrunk = (1.0 - w) * S + w * np.diag(np.diag(S))
        self._sigma_chol = np.linalg.cholesky(sigma_shrunk + _JITTER * np.eye(n_age))
        return self

    # ------------------------------------------------------- coefficient draws
    def _sample_coefs(self, n: int, rng: np.random.Generator):
        """One coefficient draw per path per equation -> banded storage.

        Returns (c, B): intercepts [n, ages] and lag coefficients
        B[path, i, k] on y_{t-1} of age i + (k - W), zero outside the range.
        """
        W = self.W
        c = np.empty((n, self.n_age))
        B = np.zeros((n, self.n_age, 2 * W + 1))

        def _draw_into(idx: np.ndarray) -> None:
            m = len(idx)
            for i, (lo, hi) in enumerate(self._bands):
                p = self._beta[i].shape[0]
                d = self._beta[i] + rng.standard_normal((m, p)) @ self._beta_chol[i].T
                c[idx, i] = d[:, 0]
                B[idx, i, lo - i + W: hi - i + W + 1] = d[:, 1:]

        _draw_into(np.arange(n))
        # Addendum 3 §7: reject explosive draws (companion spectral radius
        # >= 1) and redraw. OLS coefficient draws carry no stationarity
        # constraint; on real panels unstable draws produced predictive m_x
        # up to 3e45. Rates are NEVER clipped instead — clipping would turn
        # divergence into a bounded interval and flatter SVAR's coverage.
        for _ in range(100):
            bad = np.flatnonzero(self._spectral_radius(B) >= 1.0)
            if bad.size == 0:
                return c, B
            _draw_into(bad)
        raise ValueError(
            f"{bad.size}/{n} coefficient draws remain explosive after 100 redraws")

    def _spectral_radius(self, B: np.ndarray) -> np.ndarray:
        """|lambda|_max of each path's banded VAR(1) matrix, [n]."""
        W = self.W
        n = B.shape[0]
        A = np.zeros((n, self.n_age, self.n_age))
        for k in range(2 * W + 1):
            d = k - W
            if d < 0:
                idx = np.arange(-d, self.n_age)
                A[:, idx, idx + d] = B[:, -d:, k]
            else:
                idx = np.arange(0, self.n_age - d)
                A[:, idx, idx + d] = B[:, : self.n_age - d if d else self.n_age, k]
        return np.abs(np.linalg.eigvals(A)).max(axis=1)

    @staticmethod
    def _banded_dot(B: np.ndarray, y: np.ndarray, W: int) -> np.ndarray:
        """A @ y for banded A stored as offsets: [n, ages, 2W+1] x [n, ages]."""
        out = np.zeros_like(y)
        for k in range(2 * W + 1):
            d = k - W                                    # neighbour offset
            if d < 0:
                out[:, -d:] += B[:, -d:, k] * y[:, :d]
            elif d == 0:
                out += B[:, :, k] * y
            else:
                out[:, :-d] += B[:, :-d, k] * y[:, d:]
        return out

    # ------------------------------------------------------------- sampling
    def sample_mx(self, h: int, n: int, rng: np.random.Generator) -> np.ndarray:
        """[n, h, n_ages] predictive m_x paths.

        Coefficients drawn once per path (parameter uncertainty); a fresh
        joint innovation from the shrunk residual covariance at every horizon
        (aleatoric noise); log m integrated from the last observed year.
        """
        c, B = self._sample_coefs(n, rng)
        y = np.broadcast_to(self.y_last, (n, self.n_age)).copy()
        logm = np.broadcast_to(self.logm_last, (n, self.n_age)).copy()
        out = np.empty((n, h, self.n_age))
        for step in range(h):
            eps = rng.standard_normal((n, self.n_age)) @ self._sigma_chol.T
            y = c + self._banded_dot(B, y, self.W) + eps
            logm = logm + y                              # cumulative summation
            out[:, step, :] = logm
        return np.exp(out)

    # -------------------------------------------------------------- fitted
    def fitted_mx(self, how: str = "one_step") -> np.ndarray:
        """In-sample fitted m_x surface, [n_ages, n_years].

        The VAR models IMPROVEMENTS, so a fitted LEVEL surface needs a
        convention for where integration restarts. ``how`` selects it:

        * ``"one_step"`` (default; what the Poisson bootstrap resamples from):
          the standard VAR fitted value, the one-step-ahead conditional mean
              log m_hat[t] = log m[t-1] + c + A y[t-1],
          i.e. the OBSERVED lagged level plus the fitted improvement. It sits
          one innovation from the data at every cell, so a pseudo-panel
          D_b ~ Poisson(E * m_hat) retains the improvement variance from which
          each refit estimates Sigma_hat.
        * ``"cumulative"``: integrate the fitted improvements once from the
          first observed years, log m_hat[t] = log m[1] + sum_{s<=t} yhat[s].
          OLS-with-intercept residuals sum to zero, so this returns to the
          observed level exactly at the last year, but in between it drifts
          by the cumulated residuals (a random walk, sd ~ sqrt(t) * sigma):
          smooth, far from the data mid-sample, and a pseudo-panel built on
          it carries essentially NO improvement noise, which would collapse
          the refits' innovation covariance. Diagnostic use only; no UQ
          wrapper consumes it.

        Years 0 and 1 have no fitted improvement under either convention (the
        first improvement, year 0 -> 1, has no lag to regress on) and return
        the OBSERVED rates.
        """
        logm = self._logm
        Y = np.diff(logm, axis=1)                        # observed improvements [ages, T-1]
        Yhat = self.c_[:, None] + self.A_ @ Y[:, :-1]    # fitted improvements, years 2..T-1
        out = logm.copy()
        if how == "one_step":
            out[:, 2:] = logm[:, 1:-1] + Yhat
        elif how == "cumulative":
            out[:, 2:] = logm[:, [1]] + np.cumsum(Yhat, axis=1)
        else:
            raise ValueError(f"how must be 'one_step' or 'cumulative', got {how!r}")
        return np.exp(out)
