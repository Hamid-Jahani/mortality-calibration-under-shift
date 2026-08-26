"""Lee-Carter (1992) via SVD and Poisson Lee-Carter (Brouhns et al. 2002).

Both expose the single study-wide interface (methodology rule 4):

    model.fit(D, E)                  # [n_ages, n_years] death counts & exposures
    model.sample_mx(h, n, rng)       # -> [n, h, n_ages] predictive m_x samples

Predictive uncertainty here is the MODEL-NATIVE mechanism: random-walk-with-
drift on k_t including drift-estimation uncertainty (Lee & Carter's own
interval logic). Bootstrap / ensemble / conformal wrappers live elsewhere so
that model family x UQ mechanism stay crossed factors, never conflated.

Identifiability: sum(beta) = 1, sum(kappa) = 0 (the Lee-Carter convention).

Zero cells (PREREGISTRATION-ADDENDUM-3):
* zero-DEATH cells (E > 0, D = 0) enter on the half-count log-rate scale
  log(max(D, 0.5)/E) — addendum 2 §2's convention, replacing a 1e-10 rate
  floor whose log(1e-10) = -23.03 poisoned initial values (PLC's first Newton
  step overshot to +1e4 on ISL/LUX panels);
* zero-EXPOSURE cells (E = 0, D = 0; structural — nobody alive) carry weight
  1{E > 0} (addendum 3 §1). Under the Poisson likelihood their contribution
  is identically zero, so the weighted MLE equals the registered objective's
  MLE exactly; for the SVD stage the cells are missing entries handled by an
  EM loop that reduces to the plain SVD when no cell is missing.
"""
from __future__ import annotations

import numpy as np


def _log_rate_panel(D: np.ndarray, E: np.ndarray) -> np.ndarray:
    """log(max(D, 0.5)/E) where E > 0, NaN where E == 0 (structural zeros)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(E > 0, np.maximum(D, 0.5) / np.where(E > 0, E, 1.0), np.nan)
    return np.log(r)


class _KtForecaster:
    """Random walk with drift on k_t; drift uncertainty included.

    k_{T+h} | history ~ N( k_T + h*mu_hat,  h*sigma2 + h^2 * se_mu^2 )
    simulated pathwise so joint (path) coverage is well-defined.
    """

    def fit(self, kappa: np.ndarray) -> "_KtForecaster":
        dk = np.diff(kappa)
        self.k_last = float(kappa[-1])
        self.mu = float(dk.mean())
        # ddof=1: T-1 increments, estimating mean; Lee-Carter appendix convention
        self.sigma = float(dk.std(ddof=1))
        self.se_mu = self.sigma / np.sqrt(len(dk))
        return self

    def sample_paths(self, h: int, n: int, rng: np.random.Generator) -> np.ndarray:
        """[n, h] simulated k_t paths with drift + innovation uncertainty."""
        mu = rng.normal(self.mu, self.se_mu, size=(n, 1))          # epistemic: drift
        eps = rng.normal(0.0, self.sigma, size=(n, h))             # aleatoric: innovations
        return self.k_last + np.cumsum(mu + eps, axis=1)


class LeeCarterSVD:
    """Classical two-stage Lee-Carter: SVD on centred log m_x, then RWD on k_t."""

    #: EM iterations for missing (E = 0) cells; exact SVD when none are missing.
    _EM_ITER = 60
    _EM_TOL = 1e-10

    def fit(self, D: np.ndarray, E: np.ndarray) -> "LeeCarterSVD":
        logm = _log_rate_panel(D, E)
        miss = ~np.isfinite(logm)
        if not miss.any():
            self.alpha, self.beta, self.kappa = self._svd_stage(logm)
        else:
            # EM over the missing entries: impute with the current rank-1
            # fit, re-run the SVD stage, repeat. Missing cells are structural
            # zeros carrying no information, so the imputation only has to
            # keep the SVD numerically defined; the fit is driven entirely by
            # the observed cells. Reduces to the plain SVD at zero missing.
            work = logm.copy()
            row_mean = np.nanmean(logm, axis=1)
            work[miss] = np.broadcast_to(row_mean[:, None], logm.shape)[miss]
            prev = None
            for _ in range(self._EM_ITER):
                alpha, beta, kappa = self._svd_stage(work)
                fitted = alpha[:, None] + np.outer(beta, kappa)
                work[miss] = fitted[miss]
                if prev is not None and np.max(np.abs(fitted - prev)) < self._EM_TOL:
                    break
                prev = fitted
            self.alpha, self.beta, self.kappa = alpha, beta, kappa
        self.kt = _KtForecaster().fit(self.kappa)
        return self

    @staticmethod
    def _svd_stage(logm: np.ndarray):
        alpha = logm.mean(axis=1)                                   # [ages]
        A = logm - alpha[:, None]
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        beta, kappa = U[:, 0], s[0] * Vt[0]
        scale = beta.sum()                                          # sum(beta)=1
        beta, kappa = beta / scale, kappa * scale
        kappa = kappa - kappa.mean()                                # sum(kappa)=0
        return alpha, beta, kappa

    def sample_mx(self, h: int, n: int, rng: np.random.Generator) -> np.ndarray:
        k = self.kt.sample_paths(h, n, rng)                        # [n, h]
        return np.exp(self.alpha[None, None, :] + k[:, :, None] * self.beta[None, None, :])

    def fitted_mx(self) -> np.ndarray:
        """In-sample fitted m_x surface exp(alpha_x + beta_x kappa_t), [n_ages, n_years]."""
        return np.exp(self.alpha[:, None] + np.outer(self.beta, self.kappa))


class PoissonLeeCarter:
    """Brouhns, Denuit & Vermunt (2002): D_xt ~ Poisson(E_xt * exp(a_x + b_x k_t)).

    Fitted by the alternating Newton updates of their Section 4 (uni-dimensional
    updates of a, k, b in turn against the Poisson likelihood).
    """

    def __init__(self, max_iter: int = 500, tol: float = 1e-8):
        self.max_iter, self.tol = max_iter, tol

    def fit(self, D: np.ndarray, E: np.ndarray) -> "PoissonLeeCarter":
        n_age, n_year = D.shape
        # Initial alpha from observed cells only (nanmean over E > 0 columns);
        # half-count keeps zero-death cells off the -23 rate floor. The Newton
        # updates below need no explicit weights: at an E = 0 cell both D and
        # Dh = E exp(eta) are identically zero, so every numerator and
        # denominator contribution vanishes on its own (addendum 3 §1).
        a = np.nanmean(_log_rate_panel(D, E), axis=1)
        if not np.isfinite(a).all():
            raise ValueError("some age has no observed (E > 0) training cell")
        b = np.full(n_age, 1.0 / n_age)
        k = np.zeros(n_year)
        ll_prev = -np.inf
        for _ in range(self.max_iter):
            Dh = E * np.exp(a[:, None] + np.outer(b, k))
            a += (D - Dh).sum(axis=1) / Dh.sum(axis=1)
            Dh = E * np.exp(a[:, None] + np.outer(b, k))
            k += ((D - Dh) * b[:, None]).sum(axis=0) / (Dh * (b ** 2)[:, None]).sum(axis=0)
            k -= k.mean()
            Dh = E * np.exp(a[:, None] + np.outer(b, k))
            b += ((D - Dh) * k[None, :]).sum(axis=1) / (Dh * (k ** 2)[None, :]).sum(axis=1)
            scale = b.sum()
            b, k = b / scale, k * scale
            Dh = E * np.exp(a[:, None] + np.outer(b, k))
            # D = 0 cells contribute -Dh only (0 at E = 0 cells): 0*log(0) is
            # the likelihood's zero, not NaN.
            ll = float((np.where(D > 0, D * np.log(np.where(D > 0, Dh, 1.0)), 0.0) - Dh).sum())
            if abs(ll - ll_prev) < self.tol * (abs(ll_prev) + 1e-12):
                break
            ll_prev = ll
        self.alpha, self.beta, self.kappa = a, b, k
        self.kt = _KtForecaster().fit(k)
        return self

    def sample_mx(self, h: int, n: int, rng: np.random.Generator) -> np.ndarray:
        k = self.kt.sample_paths(h, n, rng)
        return np.exp(self.alpha[None, None, :] + k[:, :, None] * self.beta[None, None, :])

    def fitted_mx(self) -> np.ndarray:
        """In-sample fitted m_x surface exp(alpha_x + beta_x kappa_t), [n_ages, n_years]."""
        return np.exp(self.alpha[:, None] + np.outer(self.beta, self.kappa))

    def sample_deaths(self, E_future: np.ndarray, h: int, n: int,
                      rng: np.random.Generator) -> np.ndarray:
        """Full predictive death counts: Poisson noise on top of k_t paths."""
        lam = self.sample_mx(h, n, rng) * E_future[None, :h, :]
        return rng.poisson(lam).astype(float)
