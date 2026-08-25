"""Semiparametric Poisson bootstrap UQ wrapper (Brouhns, Denuit & Van Keilegom 2005).

Model-agnostic wrapper implementing the single study-wide interface
(methodology rule 4):

    PoissonBootstrap(ModelCls).fit(D, E).sample_mx(h, n, rng)  # [n, h, n_ages]

Mechanism — Brouhns, Denuit & Van Keilegom (2005), "Bootstrapping the Poisson
log-bilinear model for mortality forecasting", Scandinavian Actuarial Journal:
fit the wrapped model once on the OBSERVED deaths, form the fitted death
surface D_hat = E * m_hat, then for b = 1..B draw a pseudo-sample
D_b ~ Poisson(D_hat) and refit the model class on (D_b, E). Semiparametric
means resampling from the FITTED means — the parametric Poisson error law
around the fitted model — not residual or pairs resampling of the raw data.

Why the wrapper adds what model-native intervals lack: each refit's own
``sample_mx`` already carries the model-native forecast uncertainty (e.g.
drift + innovation noise of a random walk on k_t), while the dispersion of
parameter estimates ACROSS refits injects the sampling (estimation)
uncertainty in alpha, beta and the fitted k_t history that the native
mechanism conditions away. Pooled paths therefore embed both sources, and
every draw is a full pathwise trajectory so joint (path) coverage stays
well-defined (hypothesis H3).

Contract for the wrapped class: ``fit(D, E)``, ``sample_mx(h, n, rng)``, plus
``fitted_mx() -> [n_ages, n_years]`` (the in-sample fitted m_x surface).
UQ mechanism and model family stay crossed factors: this file knows nothing
about any particular model's internals.
"""
from __future__ import annotations

import numpy as np


class PoissonBootstrap:
    """Semiparametric Poisson bootstrap around a fitted-model CLASS.

    Parameters
    ----------
    model_cls : type
        Model class (not instance) exposing fit / sample_mx / fitted_mx.
        A fresh instance is constructed for the base fit and every refit.
    B : int
        Number of bootstrap refits. Study default 200; tests use 30.
    n_inner : int
        Reference paths per refit of the classical pooled scheme, which pools
        B * n_inner paths in total. ``sample_mx`` honours whatever ``n`` it is
        asked for by assigning paths to refits uniformly at random;
        n = B * n_inner reproduces the classical pool in expectation.
    **model_kwargs
        Forwarded to ``model_cls`` at every (re)construction.
    """

    def __init__(self, model_cls, B: int = 200, n_inner: int = 10, **model_kwargs):
        self.model_cls = model_cls
        self.B = int(B)
        self.n_inner = int(n_inner)
        self.model_kwargs = model_kwargs

    def fit(self, D: np.ndarray, E: np.ndarray,
            rng: np.random.Generator | None = None) -> "PoissonBootstrap":
        """Base fit on (D, E), then B refits on D_b ~ Poisson(E * fitted_mx).

        ``rng`` is optional so the study-wide ``fit(D, E)`` signature works
        unchanged; it controls only the resampling stage. The default is a
        fixed-seed generator so a given (D, E) always yields the same
        bootstrap world (methodology rule 7: seeds recorded).
        """
        rng = rng if rng is not None else np.random.default_rng(20260825)
        self.base = self.model_cls(**self.model_kwargs).fit(D, E)
        D_hat = np.asarray(E, dtype=float) * self.base.fitted_mx()
        self.refits = [
            self.model_cls(**self.model_kwargs).fit(rng.poisson(D_hat).astype(float), E)
            for _ in range(self.B)
        ]
        self.n_ages = int(D.shape[0])
        return self

    def sample_mx(self, h: int, n: int, rng: np.random.Generator) -> np.ndarray:
        """[n, h, n_ages] paths pooled over refits drawn uniformly with replacement.

        Two-stage draw from the bootstrap predictive mixture: refit index
        ~ Uniform{0..B-1} (parameter/estimation uncertainty), then one
        pathwise draw from that refit's own predictive law (the model-native
        drift + innovation uncertainty). Sample order groups by refit; the
        sample axis is exchangeable for every scoring rule in mortcal.eval.
        """
        counts = np.bincount(rng.integers(0, self.B, size=n), minlength=self.B)
        out = np.empty((n, h, self.n_ages))
        pos = 0
        for b, c in enumerate(counts):
            if c:
                out[pos:pos + c] = self.refits[b].sample_mx(h, int(c), rng)
                pos += c
        return out
