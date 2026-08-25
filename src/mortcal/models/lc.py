"""Lee-Carter (1992) via SVD and Poisson Lee-Carter (Brouhns et al. 2002).

Both expose the single study-wide interface (methodology rule 4):

    model.fit(D, E)                  # [n_ages, n_years] death counts & exposures
    model.sample_mx(h, n, rng)       # -> [n, h, n_ages] predictive m_x samples

Predictive uncertainty here is the MODEL-NATIVE mechanism: random-walk-with-
drift on k_t including drift-estimation uncertainty (Lee & Carter's own
interval logic). Bootstrap / ensemble / conformal wrappers live elsewhere so
that model family x UQ mechanism stay crossed factors, never conflated.

Identifiability: sum(beta) = 1, sum(kappa) = 0 (the Lee-Carter convention).
"""
from __future__ import annotations

import numpy as np


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

    def fit(self, D: np.ndarray, E: np.ndarray) -> "LeeCarterSVD":
        mx = np.clip(D / E, 1e-10, None)
        logm = np.log(mx)
        self.alpha = logm.mean(axis=1)                              # [ages]
        A = logm - self.alpha[:, None]
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        beta, kappa = U[:, 0], s[0] * Vt[0]
        scale = beta.sum()                                          # sum(beta)=1
        beta, kappa = beta / scale, kappa * scale
        kappa = kappa - kappa.mean()                                # sum(kappa)=0
        self.beta, self.kappa = beta, kappa
        self.kt = _KtForecaster().fit(kappa)
        return self

    def sample_mx(self, h: int, n: int, rng: np.random.Generator) -> np.ndarray:
        k = self.kt.sample_paths(h, n, rng)                        # [n, h]
        return np.exp(self.alpha[None, None, :] + k[:, :, None] * self.beta[None, None, :])


class PoissonLeeCarter:
    """Brouhns, Denuit & Vermunt (2002): D_xt ~ Poisson(E_xt * exp(a_x + b_x k_t)).

    Fitted by the alternating Newton updates of their Section 4 (uni-dimensional
    updates of a, k, b in turn against the Poisson likelihood).
    """

    def __init__(self, max_iter: int = 500, tol: float = 1e-8):
        self.max_iter, self.tol = max_iter, tol

    def fit(self, D: np.ndarray, E: np.ndarray) -> "PoissonLeeCarter":
        n_age, n_year = D.shape
        mx = np.clip(D / E, 1e-10, None)
        a = np.log(mx).mean(axis=1)
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
            ll = float((D * np.log(Dh) - Dh).sum())
            if abs(ll - ll_prev) < self.tol * (abs(ll_prev) + 1e-12):
                break
            ll_prev = ll
        self.alpha, self.beta, self.kappa = a, b, k
        self.kt = _KtForecaster().fit(k)
        return self

    def sample_mx(self, h: int, n: int, rng: np.random.Generator) -> np.ndarray:
        k = self.kt.sample_paths(h, n, rng)
        return np.exp(self.alpha[None, None, :] + k[:, :, None] * self.beta[None, None, :])

    def sample_deaths(self, E_future: np.ndarray, h: int, n: int,
                      rng: np.random.Generator) -> np.ndarray:
        """Full predictive death counts: Poisson noise on top of k_t paths."""
        lam = self.sample_mx(h, n, rng) * E_future[None, :h, :]
        return rng.poisson(lam).astype(float)
