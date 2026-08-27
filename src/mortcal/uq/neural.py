"""Deep ensemble (M=10) and MC dropout — the two neural-only mechanisms.

GRID.md marks both inadmissible for every classical family (deterministic
fits have no seed variance) and the ensemble inadmissible for the GP (its
posterior already integrates parameter uncertainty; the (GP, ensemble) cell
is secondary). Seeds are recorded per member (methodology rule 7).
"""
from __future__ import annotations

import numpy as np


class DeepEnsemble:
    """Lakshminarayanan et al. (2017): M independently-initialised retrains.

    Member m's torch seed derives from ``SeedSequence([seed, m])`` and is
    recorded in ``member_seeds``. ``sample_mx`` distributes the n requested
    paths over members uniformly with replacement and draws each path from
    the member's own ``sample_mx`` — a mixture of (near-)deltas for point
    families, a mixture of NB laws for NBHead. Stated in advance (spec):
    mixture-of-deltas ensembles are narrow; if they under-cover, that is a
    finding about the mechanism (H2), not a defect to widen away.
    """

    def __init__(self, model_cls, M: int = 10, seed: int = 20260825,
                 model_kwargs: dict | None = None):
        self.model_cls = model_cls
        self.M = int(M)
        self.seed = int(seed)
        self.model_kwargs = dict(model_kwargs or {})

    def fit(self, D: np.ndarray, E: np.ndarray) -> "DeepEnsemble":
        self.member_seeds = [
            int(np.random.SeedSequence([self.seed, m]).generate_state(1)[0])
            for m in range(self.M)
        ]
        self.members = [
            self.model_cls(**self.model_kwargs, seed=s).fit(D, E)
            for s in self.member_seeds
        ]
        self.n_ages = int(D.shape[0])
        return self

    def sample_mx(self, h: int, n: int, rng: np.random.Generator) -> np.ndarray:
        counts = np.bincount(rng.integers(0, self.M, size=n), minlength=self.M)
        out = np.empty((n, h, self.n_ages))
        pos = 0
        for m, c in enumerate(counts):
            if c:
                out[pos:pos + c] = self.members[m].sample_mx(h, int(c), rng)
                pos += c
        return out

    def fitted_mx(self) -> np.ndarray:
        return np.mean([m.fitted_mx() for m in self.members], axis=0)


class MCDropout:
    """Gal & Ghahramani (2016): stochastic forward passes, dropout ACTIVE.

    Requires the wrapped family to expose ``mc_sample_mx`` (families with
    dropout layers: NeuralLC, CNNLC, LSTMKt, NBHead). Families without —
    the GP, every classical family — are refused, matching GRID.md.
    """

    def __init__(self, model_cls, seed: int = 20260825,
                 model_kwargs: dict | None = None):
        self.model_cls = model_cls
        self.seed = int(seed)
        self.model_kwargs = dict(model_kwargs or {})

    def fit(self, D: np.ndarray, E: np.ndarray) -> "MCDropout":
        if not hasattr(self.model_cls, "mc_sample_mx"):
            raise ValueError(
                f"MC dropout is inadmissible for {self.model_cls.__name__}: "
                "no dropout layers (GRID.md)")
        self.model = self.model_cls(**self.model_kwargs, seed=self.seed).fit(D, E)
        self.n_ages = int(D.shape[0])
        return self

    def sample_mx(self, h: int, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.model.mc_sample_mx(h, n, rng)

    def fitted_mx(self) -> np.ndarray:
        return self.model.fitted_mx()
