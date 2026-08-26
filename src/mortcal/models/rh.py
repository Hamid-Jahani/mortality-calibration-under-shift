"""Renshaw-Haberman cohort model, simplified M2-A variant (Renshaw & Haberman 2006).

    log mu(x,t) = a_x + b_x k_t + b2_x g_{t-x},   D_xt ~ Poisson(E_xt mu(x,t))

M2-A simplification: b2_x = 1/n_ages, a CONSTANT cohort loading. Estimating a
free b2_x is a documented non-goal: the full M2 likelihood is nearly flat along
joint (b2, g) directions and its fitting is notoriously unstable (Hunt &
Villegas 2015, "Robustness and convergence in the Lee-Carter model with cohort
effects"); the constant-loading variant is the one with tolerable convergence
behaviour and is the form pre-registered as "APC / RH (M2-A)".

Interface (methodology rule 4): fit(D, E) with [n_ages, n_years] arrays, then
sample_mx(h, n, rng) -> [n, h, n_ages] predictive m_x samples, simulated
PATHWISE so joint path coverage is well-defined.

Predictive uncertainty here is the MODEL-NATIVE mechanism, matching lc.py's
convention: process uncertainty on the period index (RWD with drift-estimation
uncertainty, via _KtForecaster) and on the cohort continuation (AR(1) with
OLS-coefficient uncertainty). Parameter-estimation noise in (a, b, g) is UQ-
mechanism territory (bootstrap/ensemble wrappers), not duplicated here.

Identifiability — the constraint set, applied every iteration:

  1. sum(beta) = 1                    (exact invariance: b -> b*s, k -> k/s)
  2. sum(kappa) = 0                   (exact: a_x -> a_x + b_x * mean(k))
  3. gamma orthogonal to {1, c, c^2} over the retained cohorts (OLS fit of
     phi0 + phi1*c + phi2*c^2 subtracted every iteration), compensated by
        a_x  -> a_x + b2*(phi0 - phi1*x + phi2*x^2)
        k_t  -> k_t + phi1*t + phi2*t^2.
     Level and linear trend are the standard cohort-model gauge freedoms; the
     QUADRATIC is removed because it is a near-invariance of RH — with k_t
     near-linear in t and b_x near-constant, g_c + phi2*c^2 is compensable via
     c^2 = t^2 - 2xt + x^2 to within terms the data barely identify, so noise
     otherwise fills that direction (the approximate-identifiability pathology
     of Hunt & Villegas 2015; the same reason M7-style cohort constraints also
     remove the quadratic, Cairns et al. 2009). The a-compensation is exact;
     the k-compensation is exact only when b_x == 1/n_ages (Hunt & Villegas's
     approximate transformation, as in StMoMo's rh()); the residual misfit,
     including the -2xt cross term, is re-absorbed by the next Newton cycle
     and vanishes at the constrained optimum as the phi's -> 0.

Cohort handling: cohorts observed fewer than `min_cohort_obs` (default 5)
times are EXCLUDED from estimation — their cells get weight 0, the StMoMo
"clip" convention, so sparse diagonals never contaminate a, b, k either — and
are imputed at the fitted linear trend of the retained gamma (identically ~0
after constraint 3) for forecasting.

Fitting: alternating one-dimensional Newton updates against the Poisson
likelihood in the style of Brouhns, Denuit & Vermunt (2002) / lc.py's
PoissonLeeCarter, with damped steps (factor 0.5 on a likelihood decrease) and
a hard iteration cap — RH pathologies are a known schedule risk, so the fit
never loops forever and exposes a `.converged` flag instead.
"""
from __future__ import annotations

import numpy as np

from .lc import _KtForecaster


class _CohortAR1:
    """AR(1) with intercept for the cohort-index continuation.

        g_{c+1} = c0 + phi * g_c + eps,   eps ~ N(0, sigma^2)

    Fitted by OLS on consecutive retained cohorts. Because constraint 3 removes
    the linear drift of gamma into (a, k), this is "AR(1) around drift" with the
    drift identically zero by construction; c0 and phi are still estimated
    freely. Parameter uncertainty: (c0, phi) sampled per path from the OLS
    sampling covariance (the analogue of _KtForecaster's se_mu); innovation
    noise simulated pathwise.
    """

    def fit(self, g: np.ndarray) -> "_CohortAR1":
        y, x = g[1:], g[:-1]
        X = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        dof = max(len(y) - 2, 1)
        self.sigma = float(np.sqrt(resid @ resid / dof))
        self.coef = coef                                        # [c0, phi]
        self.coef_cov = np.linalg.inv(X.T @ X) * self.sigma ** 2
        self.g_last = float(g[-1])
        return self

    def sample_paths(self, h: int, n: int, rng: np.random.Generator) -> np.ndarray:
        """[n, h] simulated continuations of g from the last retained cohort."""
        coefs = rng.multivariate_normal(self.coef, self.coef_cov, size=n)
        c0 = coefs[:, 0]
        phi = np.clip(coefs[:, 1], -0.995, 0.995)               # keep paths stable
        out = np.empty((n, h))
        state = np.full(n, self.g_last)
        for j in range(h):
            state = c0 + phi * state + rng.normal(0.0, self.sigma, size=n)
            out[:, j] = state
        return out


class RenshawHaberman:
    """RH M2-A: Poisson Lee-Carter plus a constant-loading cohort effect."""

    def __init__(self, max_iter: int = 2000, tol: float = 1e-8,
                 min_cohort_obs: int = 5):
        self.max_iter, self.tol = max_iter, tol
        self.min_cohort_obs = min_cohort_obs

    def fit(self, D: np.ndarray, E: np.ndarray) -> "RenshawHaberman":
        n_a, n_t = D.shape
        b2 = 1.0 / n_a
        x_idx = np.arange(n_a, dtype=float)
        t_idx = np.arange(n_t, dtype=float)
        # cohort c = t - x; index shifted so idx 0 is the oldest cohort
        cidx = np.arange(n_t)[None, :] - np.arange(n_a)[:, None] + (n_a - 1)
        n_c = n_a + n_t - 1
        cvals = np.arange(n_c, dtype=float) - (n_a - 1)          # actual t - x values
        flat = cidx.ravel()

        counts = np.bincount(flat, minlength=n_c)
        retained = counts >= self.min_cohort_obs
        ridx = np.where(retained)[0]
        if len(ridx) < 10:
            raise ValueError("too few well-observed cohorts to fit M2-A")
        if not np.all(np.diff(ridx) == 1):
            raise ValueError("retained cohorts are not contiguous")
        W = retained[cidx].astype(float)                         # StMoMo clip weights
        # Addendum 3 §1: structural E = 0 cells carry weight 0 (their Poisson
        # contribution is identically zero); zero-death observed cells enter
        # the INITIALISER on the half-count scale (addendum 2 §2) instead of
        # a 1e-10 floor whose log = -23.03 wrecked the first Newton step.
        W = W * (E > 0)

        with np.errstate(divide="ignore", invalid="ignore"):
            init_rate = np.where(E > 0, np.maximum(D, 0.5) / np.where(E > 0, E, 1.0), np.nan)
        a = np.nanmean(np.log(init_rate), axis=1)
        if not np.isfinite(a).all():
            raise ValueError("some age has no observed (E > 0) training cell")
        b = np.full(n_a, 1.0 / n_a)
        k = np.zeros(n_t)
        g = np.zeros(n_c)

        def eta(a, b, k, g):
            return a[:, None] + np.outer(b, k) + b2 * g[cidx]

        def loglik(Dh):
            # W masks excluded-cohort cells; D=0 cells contribute -Dh only
            return float((W * (np.where(D > 0, D * np.log(np.where(D > 0, Dh, 1.0)), 0.0) - Dh)).sum())

        ll_prev = -np.inf
        step = 1.0
        self.converged = False
        it = 0
        while it < self.max_iter:
            it += 1
            a0, b0, k0, g0 = a.copy(), b.copy(), k.copy(), g.copy()

            # --- alternating uni-dimensional Newton updates (Brouhns-style) ---
            Dh = E * np.exp(eta(a, b, k, g))
            a = a + step * (W * (D - Dh)).sum(1) / np.maximum((W * Dh).sum(1), 1e-300)
            Dh = E * np.exp(eta(a, b, k, g))
            k = k + step * ((W * (D - Dh) * b[:, None]).sum(0)
                            / np.maximum((W * Dh * (b ** 2)[:, None]).sum(0), 1e-300))
            Dh = E * np.exp(eta(a, b, k, g))
            den_b = (W * Dh * (k ** 2)[None, :]).sum(1)
            num_b = (W * (D - Dh) * k[None, :]).sum(1)
            b = b + step * np.where(den_b > 0, num_b / np.maximum(den_b, 1e-300), 0.0)
            Dh = E * np.exp(eta(a, b, k, g))
            num_g = np.bincount(flat, (W * (D - Dh)).ravel(), minlength=n_c) * b2
            den_g = np.bincount(flat, (W * Dh).ravel(), minlength=n_c) * b2 * b2
            g = g + step * np.where(retained & (den_g > 0),
                                    num_g / np.maximum(den_g, 1e-300), 0.0)

            # --- identifiability constraints (module docstring, set 1-3) ---
            # degree-2: {1, c} are exact/near gauge freedoms; c^2 pins the
            # near-invariant quadratic direction (c^2 = t^2 - 2xt + x^2) that
            # otherwise fills with noise and bends kappa (docstring, set 3)
            phi2, phi1, phi0 = np.polyfit(cvals[ridx], g[ridx], 2)
            g = np.where(retained, g - (phi0 + phi1 * cvals + phi2 * cvals ** 2), 0.0)
            # a-compensation keeps only the x-pure terms (exact); the t-pure
            # and -2xt cross terms are left for Newton to re-absorb with
            # likelihood-optimal weights (see NOTE below on why forcing them
            # into k with b_x != b2 biases forecasts)
            a = a + b2 * (phi0 - phi1 * x_idx + phi2 * x_idx ** 2)
            # NOTE deliberately NO `k += phi1 * t` here. That textbook shortcut
            # is exact only when b_x == b2 for every age; with heterogeneous
            # beta it force-feeds the cohort trend into the period index with
            # the WRONG age-weights, and since forecasts are not gauge-
            # invariant (k is extrapolated by RWD, gamma mean-reverts), the
            # misallocated trend becomes a systematic forecast bias. Leaving
            # the b2*phi1*t component unabsorbed lets the next Newton k-update
            # restore exactly as much of it as the likelihood supports.
            m = k.mean()
            k = k - m
            a = a + b * m
            s = b.sum()
            b, k = b / s, k * s

            # --- damping and convergence bookkeeping ---
            Dh = E * np.exp(eta(a, b, k, g))
            ll = loglik(Dh)
            tol_abs = self.tol * (abs(ll_prev) + 1e-12)
            if not np.isfinite(ll) or (np.isfinite(ll_prev) and ll < ll_prev - tol_abs):
                # divergence: revert the whole cycle, halve the step factor
                a, b, k, g = a0, b0, k0, g0
                step *= 0.5
                if step < 1e-8:          # step annihilated -> give up cleanly
                    break
                continue
            if np.isfinite(ll_prev) and abs(ll - ll_prev) < tol_abs:
                ll_prev = ll
                self.converged = True
                break
            ll_prev = ll
            step = min(1.0, 2.0 * step)  # recover the step after accepted moves

        self.n_iter, self.loglik_ = it, ll_prev
        self.alpha, self.beta, self.kappa = a, b, k
        self.b2 = b2
        # Impute excluded cohorts at the fitted linear trend of the retained
        # gamma (constraint 3 makes this ~0; computed explicitly regardless).
        tr1, tr0 = np.polyfit(cvals[ridx], g[ridx], 1)
        self.gamma = np.where(retained, g, tr0 + tr1 * cvals)
        self.gamma_retained = retained
        self.cohort_index = cvals
        self.n_ages, self.n_years = n_a, n_t
        self._last_retained = int(ridx[-1])
        self.kt = _KtForecaster().fit(k)
        self.gt = _CohortAR1().fit(g[ridx])
        return self

    def sample_mx(self, h: int, n: int, rng: np.random.Generator) -> np.ndarray:
        """[n, h, n_ages] pathwise predictive m_x samples.

        Cells whose cohort is already observed use the fitted (or trend-
        imputed) gamma; cells belonging to unborn-in-sample cohorts
        (c >= n_years) use the AR(1) continuation, simulated from the last
        RETAINED cohort so the trailing excluded diagonals contribute the
        correct number of innovation steps to the predictive variance.
        """
        n_a, n_t = self.n_ages, self.n_years
        n_c = n_a + n_t - 1
        k = self.kt.sample_paths(h, n, rng)                     # [n, h]
        gap = (n_c - 1) - self._last_retained                   # trailing excluded
        sim = self.gt.sample_paths(gap + h, n, rng)             # [n, gap+h]
        g_future = sim[:, gap:]                                 # c = n_t .. n_t-1+h
        table = np.concatenate(
            [np.broadcast_to(self.gamma, (n, n_c)), g_future], axis=1)
        # forecast-cell cohort indices: year n_t-1+j, age x -> c = n_t-1+j-x
        cf = (n_t - 1 + np.arange(1, h + 1))[:, None] - np.arange(n_a)[None, :]
        g_cells = table[:, cf + (n_a - 1)]                      # [n, h, n_ages]
        eta = (self.alpha[None, None, :]
               + self.beta[None, None, :] * k[:, :, None]
               + self.b2 * g_cells)
        return np.exp(eta)

    def fitted_mx(self) -> np.ndarray:
        """In-sample fitted m_x surface exp(a_x + b_x k_t + b2 g_{t-x}), [n_ages, n_years].

        Every cell reads the fitted cohort table ``gamma``: retained cohorts
        carry their estimate; cohorts EXCLUDED from estimation (fewer than
        ``min_cohort_obs`` cells, weight 0 in the fit) carry the imputed
        linear-trend value (~0 under constraint 3). The surface is therefore
        defined on the whole rectangle — the Poisson bootstrap wrapper needs
        D_b ~ Poisson(E * fitted_mx) at every cell — but on excluded-cohort
        cells it is the model's extrapolation, not a fit to those cells.
        """
        n_a, n_t = self.n_ages, self.n_years
        cidx = np.arange(n_t)[None, :] - np.arange(n_a)[:, None] + (n_a - 1)
        eta = (self.alpha[:, None] + np.outer(self.beta, self.kappa)
               + self.b2 * self.gamma[cidx])
        return np.exp(eta)

    def sample_deaths(self, E_future: np.ndarray, h: int, n: int,
                      rng: np.random.Generator) -> np.ndarray:
        """Full predictive death counts: Poisson noise on top of the paths."""
        lam = self.sample_mx(h, n, rng) * E_future[None, :h, :]
        return rng.poisson(lam).astype(float)
