"""Multi-output GP over years with ages as tasks (docs/NEURAL-SPEC.md §5).

Huynh & Ludkovski (2021) reduced to one (population, sex) panel: an exact
multitask GP, input = scaled calendar year, tasks = ages, kernel =
RBF(year) ⊗ ICM(rank 5), Gaussian likelihood on the half-count log rates
log(max(D, 0.5)/E). The posterior over the h future years is the
model-native predictive law — GRID.md marks the bootstrap inadmissible for
this family because the posterior already integrates parameter uncertainty.

Kronecker structure needs a complete year × age block, so the GP trains on
the TRAILING block of complete years (every age observed) — the same
common-complete principle as SVAR (addendum 3 §1) applied to the trailing
window; for the affected populations every incomplete year is pre-1970.

gpytorch is OPTIONAL (uv sync --group neural).
"""
from __future__ import annotations

import numpy as np

try:  # optional — see module docstring
    import torch
    import gpytorch
except ImportError:  # pragma: no cover
    torch = None
    gpytorch = None

from contextlib import contextmanager

from .lc import _log_rate_panel


@contextmanager
def _gp_ctx():
    """Every GP linear-algebra path shares one numerics context: generous
    Cholesky jitter on float64 (the mortality task covariance is legitimately
    near-low-rank — ~100 ages driven by a handful of latent factors) and the
    fast predictive-variance cache."""
    with gpytorch.settings.fast_pred_var(),             gpytorch.settings.cholesky_jitter(double_value=1e-5),             gpytorch.settings.cholesky_max_tries(9):
        yield


class MultiOutputGP:
    """Exact multitask GP: year in, all ages out, joint posterior sampling."""

    lr_grid = (1e-1, 3e-2)
    iters_grid = (200, 400)

    def __init__(self, lr_grid=None, iters_grid=None, seed: int = 20260825,
                 rank: int = 5, min_years: int = 40,
                 inner_val_years: int = 5, max_years: int | None = None):
        if lr_grid is not None:
            self.lr_grid = tuple(lr_grid)
        if iters_grid is not None:
            self.iters_grid = tuple(iters_grid)
        self.seed = int(seed)
        self.rank = int(rank)
        self.min_years = int(min_years)
        self.inner_val_years = int(inner_val_years)
        # PREREGISTRATION-ADDENDUM-4: cap on the trailing training window.
        # Exact-GP kernel memory scales with (years x ages)^2 - a 269-year panel
        # is a 5.8 GB kernel per fit and the conformal wrappers refit ten times;
        # the production run stalled swapping. None = whole trailing block.
        self.max_years = None if max_years is None else int(max_years)

    # ------------------------------------------------------------------ fit
    def fit(self, D: np.ndarray, E: np.ndarray) -> "MultiOutputGP":
        if gpytorch is None:
            raise ImportError(
                "MultiOutputGP needs torch + gpytorch: uv sync --group neural")
        D = np.asarray(D, dtype=float)
        E = np.asarray(E, dtype=float)
        self.n_ages, T_full = D.shape

        logm = _log_rate_panel(D, E)                       # NaN at E = 0
        complete = np.isfinite(logm).all(axis=0)
        # trailing contiguous block of complete years
        incomplete = np.flatnonzero(~complete)
        start = int(incomplete[-1]) + 1 if incomplete.size else 0
        if T_full - start < self.min_years:
            raise ValueError(
                f"trailing complete block has {T_full - start} years; "
                f"MultiOutputGP needs >= {self.min_years}")
        if self.max_years is not None:                     # addendum 4 window cap
            start = max(start, T_full - self.max_years)
        self._start = start
        Y = logm[:, start:].T                              # [T, ages]
        self.T = Y.shape[0]
        self._x = torch.linspace(0.0, 1.0, self.T, dtype=torch.float64)
        self._x_step = 1.0 / max(self.T - 1, 1)
        # Standardise each task (age) before fitting and invert on sampling.
        # The ICM shares one kernel across tasks, so it assumes comparable
        # task scales; raw HMD log rates do not have them — measured on SWE
        # males 1980-2019 the level spans [-11.74, -0.37] and the per-age sd
        # ranges 0.063 (age 95) to 0.85, a 13x spread. Unstandardised, the
        # multitask covariance is numerically singular for more than ~20 ages
        # (measured: 20 ages fit, 40/60/100 raise NotPSDError at any year
        # span, including the source paper's 27). This is preprocessing, not
        # a change of model: the posterior is mapped back exactly.
        self._mu = Y.mean(axis=0)                          # [ages]
        sd = Y.std(axis=0, ddof=1)
        self._sd = np.where(sd > 1e-8, sd, 1.0)
        self._Y = torch.as_tensor((Y - self._mu) / self._sd, dtype=torch.float64)

        # tuning on the trailing inner-validation years (marginal-likelihood
        # training on the inner-train block; val = predictive NLL)
        v = self.inner_val_years
        best = None
        for lr in self.lr_grid:
            for iters in self.iters_grid:
                with _gp_ctx():
                    model, lik = self._train(self._x[:-v], self._Y[:-v], lr, iters)
                    score = self._val_nll(model, lik, self._x[-v:], self._Y[-v:])
                if best is None or score < best[0]:
                    best = (score, lr, iters)
        _, self.lr_, self.iters_ = best
        with _gp_ctx():
            self.model_, self.lik_ = self._train(self._x, self._Y, self.lr_, self.iters_)
        return self

    def _train(self, x, Y, lr, iters):
        torch.manual_seed(self.seed)
        n_tasks, rank = self.n_ages, self.rank

        class MTGP(gpytorch.models.ExactGP):
            def __init__(self, tx, ty, likelihood):
                super().__init__(tx, ty, likelihood)
                self.mean = gpytorch.means.MultitaskMean(
                    gpytorch.means.ConstantMean(), num_tasks=n_tasks)
                self.covar = gpytorch.kernels.MultitaskKernel(
                    gpytorch.kernels.RBFKernel(), num_tasks=n_tasks, rank=rank)

            def forward(self, tx):
                return gpytorch.distributions.MultitaskMultivariateNormal(
                    self.mean(tx), self.covar(tx))

        # ONE floored homoscedastic noise. The default likelihood carries a
        # global noise PLUS 100 unconstrained per-task noises; those collapsed
        # toward zero and left K + Sigma singular (measured: NotPSDError for
        # more than ~20 ages at any year span, including the source paper's
        # 27). After per-task standardisation every task has unit variance, so
        # a single shared noise is the right structure rather than a
        # concession — and the floor is substantively right too: the observed
        # log crude rate carries Poisson sampling noise, sd ~ 1/sqrt(D), so
        # zero observation noise is not a state the data can be in.
        lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=n_tasks, has_task_noise=False,
            noise_constraint=gpytorch.constraints.GreaterThan(1e-4)).double()
        model = MTGP(x, Y, lik).double()
        model.train(); lik.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
        for _ in range(int(iters)):
            opt.zero_grad()
            loss = -mll(model(x), Y)
            loss.backward()
            opt.step()
        model.eval(); lik.eval()
        return model, lik

    @staticmethod
    def _val_nll(model, lik, x_val, Y_val) -> float:
        """MARGINAL predictive NLL on the inner-validation years.

        Deliberately not the joint ``log_prob``: that needs a Cholesky of the
        full [n_val * n_tasks] predictive covariance, which is the single
        place the multitask GP was numerically fragile, and hyperparameter
        SELECTION does not need the joint density — only a consistent
        ranking of configurations. Cell-wise Gaussian NLL from the predictive
        mean and variance gives that with no factorisation at all.
        """
        with torch.no_grad():
            pred = lik(model(x_val))
            mu = pred.mean
            var = torch.clamp(pred.variance, min=1e-10)
            nll = 0.5 * (torch.log(2 * np.pi * var) + (Y_val - mu) ** 2 / var)
            return float(nll.mean())

    # ------------------------------------------------------------- sampling
    def sample_mx(self, h: int, n: int, rng: np.random.Generator) -> np.ndarray:
        """[n, h, ages] joint posterior samples at the h future years — the
        GP's own predictive law (native mechanism)."""
        x_fut = 1.0 + self._x_step * torch.arange(1, h + 1, dtype=torch.float64)
        g = torch.Generator()
        g.manual_seed(int(rng.integers(0, 2**63 - 1)))
        with torch.no_grad(), _gp_ctx():
            post = self.lik_(self.model_(x_fut))
            torch.manual_seed(int(g.initial_seed()))
            z = post.rsample(torch.Size([n])).reshape(n, h, self.n_ages)
        return np.exp(z.numpy() * self._sd[None, None, :] + self._mu[None, None, :])

    def median_logmx(self, h: int) -> np.ndarray:
        """[h, ages] pointwise median of the predictive law of log m_x — for a
        Gaussian posterior that is exactly the posterior mean, so no draws are
        needed. The conformal wrappers use this instead of estimating the
        median from ``n_median_samples`` posterior draws: on the 269-year SWE
        panel that ``rsample`` of 1000 joint draws allocated 1.59 GB and
        failed SOLO (results/timings_solo.json), which had been misread as
        memory pressure from concurrent jobs. Exact, allocation-free, and
        identical in expectation to what the sampling estimated."""
        x_fut = 1.0 + self._x_step * torch.arange(1, h + 1, dtype=torch.float64)
        with torch.no_grad(), _gp_ctx():
            mean = self.lik_(self.model_(x_fut)).mean                # [h, ages], z-scale
        return mean.numpy() * self._sd[None, :] + self._mu[None, :]

    def fitted_mx(self) -> np.ndarray:
        """Posterior-mean in-sample surface on the trained block, [ages, T]."""
        with torch.no_grad(), _gp_ctx():
            mean = self.lik_(self.model_(self._x)).mean            # [T, ages], z-scale
        return np.exp(mean.numpy() * self._sd[None, :] + self._mu[None, :]).T
