"""The four torch families of docs/NEURAL-SPEC.md.

* ``NeuralLC``  — Richman–Wüthrich (2021) embedding network, per-panel form.
* ``CNNLC``     — shallow convolutional Lee–Carter (Perla et al. 2021 /
                  Schnürch–Korn 2022 form).
* ``LSTMKt``    — LSTM on the Lee–Carter index (SVD stage reused verbatim).
* ``NBHead``    — distributional negative-binomial head.

Shared rules (spec §Ground rules): count models train on the Poisson deviance
with log(E) offset; hyperparameters tune on the last ``inner_val_years``
training years over the small registered grids; every fit is deterministic
given its ``seed``; ``W = 1{E > 0}`` cells are excluded from every loss
(addendum 3 §1) and zero-death cells enter feature panels on the half-count
scale (addendum 2 §2).

torch is an OPTIONAL dependency: importing this module without torch succeeds;
constructing a family raises with the install command. The classical grid
never waits on the CUDA wheel.
"""
from __future__ import annotations

import numpy as np

try:  # optional — see module docstring
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only in torch-less envs
    torch = None
    nn = None

from .lc import LeeCarterSVD, _log_rate_panel


def _require_torch():
    if torch is None:
        raise ImportError(
            "the neural families need torch: uv sync --group neural")


def _device() -> "torch.device":
    """Compute device for the torch families, from MORTCAL_DEVICE (default cpu).

    Default is CPU on purpose: at this model scale (2x64 hidden, <= 27k
    cells, full-batch Adam) kernel-launch overhead can exceed the arithmetic,
    so GPU is an opt-in that must be justified by a measured per-cell timing
    (results/timings_*.json), never assumed. Requesting cuda without a
    usable device is an error, not a silent fallback — a sweep must know
    what it ran on.
    """
    import os
    name = os.environ.get("MORTCAL_DEVICE", "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("MORTCAL_DEVICE=cuda but torch.cuda.is_available() is False")
    return torch.device(name)


def _torch_seed_from(rng: np.random.Generator) -> "torch.Generator":
    """A torch generator advanced from the caller's numpy generator, so the
    study-wide 'rng in, reproducible draws out' contract holds for torch
    sampling too."""
    g = torch.Generator()
    g.manual_seed(int(rng.integers(0, 2**63 - 1)))
    return g


def _poisson_deviance(f_log_mx, D, E, w):
    """Poisson deviance kernel  sum w * (mu - d log mu),  mu = E exp(f).

    The registered likelihood (PREREGISTRATION.md:40). ``w`` is the
    1{E > 0} weight — at w = 0 the cell contributes nothing, mirroring the
    exact-zero Fisher information of a structural zero (addendum 3 §1).
    """
    log_mu = f_log_mx + torch.log(torch.clamp(E, min=1.0))
    return (w * (torch.exp(log_mu) - D * log_mu)).sum() / w.sum()


class _TorchFamily:
    """Shared fit/tune scaffolding.

    Subclasses implement ``_build(n_ages)`` -> nn.Module, ``_tensors(D, E)``
    -> training tensors, ``_loss(net, tensors, idx)`` and
    ``_point_logmx(h)``. Tuning: for each (lr, epochs) in the registered
    grids, train from the SAME seeded init on the inner-train cells, score
    the Poisson deviance on the inner-validation cells (last
    ``inner_val_years`` years), pick the arg-min, refit on the full window.
    """

    #: overridden per family with the registered grids (spec + tests).
    lr_grid: tuple = ()
    epochs_grid: tuple = ()

    def __init__(self, lr_grid=None, epochs_grid=None, seed: int = 20260825,
                 inner_val_years: int = 5, dropout: float = 0.05):
        if lr_grid is not None:
            self.lr_grid = tuple(lr_grid)
        if epochs_grid is not None:
            self.epochs_grid = tuple(epochs_grid)
        self.seed = int(seed)
        self.inner_val_years = int(inner_val_years)
        self.dropout = float(dropout)

    # ---- template ---------------------------------------------------------
    def fit(self, D: np.ndarray, E: np.ndarray):
        _require_torch()
        D = np.asarray(D, dtype=float)
        E = np.asarray(E, dtype=float)
        self.n_ages, self.T = D.shape
        # per-fit cache of training subsets keyed by the excluded-year set:
        # the masked / stacked / paired tensors are constant within one
        # training run, so building them once instead of every epoch changes
        # nothing numerically. MEASURED 2026-08-27 (results/timings_cached.json
        # vs timings_solo.json): NO wall-time gain — per-epoch re-indexing was
        # not the dominant cost, the arithmetic is. Kept because it is free
        # and harmless, not because it is fast. The real lever is the device:
        # cuda gives 5-8x on NB/NLC/CNN (timings_gpu.json), ~1x on LSTM.
        self._subset_cache: dict = {}
        self.device = _device()
        self._prepare(D, E)
        for k, v in list(vars(self).items()):          # training tensors -> device
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(self.device))

        val_years = set(range(self.T - self.inner_val_years, self.T))
        best = None
        for lr in self.lr_grid:
            for epochs in self.epochs_grid:
                net = self._train(lr, epochs, exclude_years=val_years)
                v = float(self._val_loss(net, val_years))
                if best is None or v < best[0]:
                    best = (v, lr, epochs)
        _, self.lr_, self.epochs_ = best
        self.net_ = self._train(self.lr_, self.epochs_, exclude_years=set())
        self.net_.eval()
        return self

    def _train(self, lr, epochs, exclude_years):
        torch.manual_seed(self.seed)          # same init for every config
        net = self._build().to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        net.train()
        for _ in range(int(epochs)):
            opt.zero_grad()
            loss = self._loss(net, exclude_years)
            loss.backward()
            opt.step()
        return net

    # ---- degenerate native (spec §Ground rules 6) -------------------------
    def sample_mx(self, h: int, n: int, rng: np.random.Generator) -> np.ndarray:
        """[n, h, n_ages] — the DEGENERATE repeated point path.

        Point families have no predictive law of their own (GRID.md marks
        their native cell inadmissible). The only legitimate consumers of
        this method are the conformal centre (median of identical paths =
        the point path) and point metrics. ``run_cell`` refuses the native
        cell before any fit.
        """
        pt = np.exp(self._point_logmx(h))                   # [h, ages]
        return np.broadcast_to(pt, (n, h, self.n_ages)).copy()

    def fitted_mx(self) -> np.ndarray:
        """In-sample fitted m_x surface [n_ages, T] (pboot hook)."""
        with torch.no_grad():
            return np.exp(self._insample_logmx())


# ---------------------------------------------------------------------------
# 1. NeuralLC — Richman–Wüthrich embeddings, per-panel form
# ---------------------------------------------------------------------------

class _CellNet:
    """Builds the (age-embedding ⊕ year) trunk shared by NeuralLC / NBHead."""

    @staticmethod
    def build(n_ages: int, embed_dim: int, hidden: int, dropout: float,
              n_out: int, bias0: float = 0.0) -> "nn.Module":
        """``bias0`` centres the first output at the empirical mean log rate,
        so training starts at the mean surface instead of log m = 0 (m = 1) —
        plain centring, not tuning: without it the registered epoch budgets
        spend most of their steps travelling to the data's scale."""
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(n_ages, embed_dim)
                self.trunk = nn.Sequential(
                    nn.Linear(embed_dim + 1, hidden), nn.Tanh(), nn.Dropout(dropout),
                    nn.Linear(hidden, hidden), nn.Tanh(), nn.Dropout(dropout),
                    nn.Linear(hidden, n_out),
                )
                with torch.no_grad():
                    self.trunk[-1].bias[0] = float(bias0)

            def forward(self, age_idx, year_x):
                z = torch.cat([self.emb(age_idx), year_x[:, None]], dim=1)
                return self.trunk(z)

        return Net()


class _CellFamily(_TorchFamily):
    """Cell-feature families (NeuralLC, NBHead): one row per (age, year)."""

    n_out = 1

    def __init__(self, lr_grid=None, epochs_grid=None, seed: int = 20260825,
                 inner_val_years: int = 5, dropout: float = 0.05,
                 embed_dim: int = 5, hidden: int = 64):
        super().__init__(lr_grid, epochs_grid, seed, inner_val_years, dropout)
        self.embed_dim, self.hidden = int(embed_dim), int(hidden)

    def _prepare(self, D, E):
        aa, tt = np.meshgrid(np.arange(self.n_ages), np.arange(self.T),
                             indexing="ij")
        self._age = torch.as_tensor(aa.ravel(), dtype=torch.long)
        self._yr_idx = tt.ravel()
        self._yearx = torch.as_tensor(self._scale_year(self._yr_idx),
                                      dtype=torch.float32)
        self._D = torch.as_tensor(D.ravel(), dtype=torch.float32)
        self._E = torch.as_tensor(E.ravel(), dtype=torch.float32)
        self._w = torch.as_tensor((E.ravel() > 0).astype("float32"))
        obs = E.ravel() > 0
        self._bias0 = float(np.log(max(D.ravel()[obs].sum(), 0.5) / E.ravel()[obs].sum()))

    def _scale_year(self, t):
        """Affine map: last training year -> 0, first -> about -1; future > 0."""
        return (np.asarray(t, dtype=float) - (self.T - 1)) / max(self.T - 1, 1)

    def _build(self):
        return _CellNet.build(self.n_ages, self.embed_dim, self.hidden,
                              self.dropout, self.n_out, bias0=self._bias0)

    def _mask(self, exclude_years):
        keep = ~np.isin(self._yr_idx, list(exclude_years)) if exclude_years else \
            np.ones_like(self._yr_idx, dtype=bool)
        return torch.as_tensor(keep).to(self.device)

    def _subset(self, exclude_years):
        """(age, yearx, D, E, w) restricted to the kept cells — built once per
        excluded-year set and cached for the whole training run."""
        key = frozenset(exclude_years)
        sub = self._subset_cache.get(key)
        if sub is None:
            m = self._mask(exclude_years)
            sub = (self._age[m], self._yearx[m], self._D[m], self._E[m], self._w[m])
            self._subset_cache[key] = sub
        return sub

    def _loss(self, net, exclude_years):
        age, yearx, D, E, w = self._subset(exclude_years)
        return self._nll(net(age, yearx), D, E, w)

    def _val_loss(self, net, val_years):
        net.eval()
        with torch.no_grad():
            m = torch.as_tensor(np.isin(self._yr_idx, list(val_years))).to(self.device)
            return self._nll(net(self._age[m], self._yearx[m]),
                             self._D[m], self._E[m], self._w[m])

    def _forward_years(self, net, year_idx):
        """net outputs on the full age range at the given year indices,
        [len(year_idx), n_ages, n_out]."""
        yrs = np.asarray(year_idx)
        aa, tt = np.meshgrid(np.arange(self.n_ages), yrs, indexing="ij")
        age = torch.as_tensor(aa.ravel(), dtype=torch.long).to(self.device)
        yx = torch.as_tensor(self._scale_year(tt.ravel()),
                             dtype=torch.float32).to(self.device)
        out = net(age, yx).reshape(self.n_ages, len(yrs), self.n_out)
        return out.permute(1, 0, 2)                          # [years, ages, out]

    def _insample_logmx(self):
        out = self._forward_years(self.net_, np.arange(self.T))
        return out[:, :, 0].T.cpu().numpy()                        # [ages, T]


class NeuralLC(_CellFamily):
    """Richman–Wüthrich (2021) embedding network, per-(pop, sex) panel form."""

    lr_grid = (1e-2, 3e-3)
    epochs_grid = (200, 500)
    n_out = 1

    def _nll(self, out, D, E, w):
        return _poisson_deviance(out[:, 0], D, E, w)

    def _point_logmx(self, h: int) -> np.ndarray:
        self.net_.eval()
        with torch.no_grad():
            out = self._forward_years(self.net_, np.arange(self.T, self.T + h))
        return out[:, :, 0].cpu().numpy()                          # [h, ages]

    def mc_sample_mx(self, h: int, n: int, rng: np.random.Generator) -> np.ndarray:
        """[n, h, ages] stochastic forward passes with dropout ACTIVE."""
        g = _torch_seed_from(rng)
        torch.manual_seed(int(g.initial_seed()))
        self.net_.train()                                    # dropout on
        outs = np.empty((n, h, self.n_ages))
        with torch.no_grad():
            for i in range(n):
                o = self._forward_years(self.net_, np.arange(self.T, self.T + h))
                outs[i] = o[:, :, 0].cpu().numpy()
        self.net_.eval()
        return np.exp(outs)


# ---------------------------------------------------------------------------
# 2. CNNLC — shallow convolutional Lee–Carter
# ---------------------------------------------------------------------------

class CNNLC(_TorchFamily):
    """One Conv2d over a trailing (L years × ages) log-rate window, linear
    head to next year's log m_x; recursive multi-step forecast.

    Grid corrected 2026-08-27 before any real-data run: the spec's original
    {1e-2, 3e-3} learning rates DIVERGE on this loss scale (measured
    in-sample RMSE 10-20 nats at every grid point; 1e-3 converges to 0.12
    and beats the true model's forecast RMSE on the synthetic DGP)."""

    lr_grid = (1e-3, 3e-4)
    epochs_grid = (300, 800)

    def __init__(self, lr_grid=None, epochs_grid=None, seed: int = 20260825,
                 inner_val_years: int = 5, dropout: float = 0.05,
                 window: int = 10, channels: int = 8):
        super().__init__(lr_grid, epochs_grid, seed, inner_val_years, dropout)
        self.L, self.channels = int(window), int(channels)

    def _prepare(self, D, E):
        logm = _log_rate_panel(D, E)                          # NaN at E = 0
        # forward-fill each age along years, then back-fill leading NaN: the
        # fill enters INPUT windows only, never a training target.
        filled = logm.copy()
        for a in range(self.n_ages):
            row = filled[a]
            miss = ~np.isfinite(row)
            if miss.any():
                idx = np.where(~miss, np.arange(self.T), -1)
                np.maximum.accumulate(idx, out=idx)
                row[:] = np.where(idx >= 0, row[np.maximum(idx, 0)], row)
                if not np.isfinite(row[0]):
                    first = np.flatnonzero(np.isfinite(row))[0]
                    row[:first] = row[first]
        self._logm_fill = filled
        self._X = torch.as_tensor(filled, dtype=torch.float32)
        self._D = torch.as_tensor(D, dtype=torch.float32)
        self._E = torch.as_tensor(E, dtype=torch.float32)
        self._w = torch.as_tensor((E > 0).astype("float32"))
        if self.T <= self.L + self.inner_val_years:
            raise ValueError(f"need > {self.L + self.inner_val_years} training "
                             f"years for CNN window {self.L}; got {self.T}")

    def _build(self):
        L, A, C = self.L, self.n_ages, self.channels

        class Net(nn.Module):
            def __init__(self, bias_by_age):
                super().__init__()
                self.conv = nn.Conv2d(1, C, kernel_size=3, padding=1)
                self.drop = nn.Dropout(0.05)
                self.head = nn.Linear(C * L * A, A)
                with torch.no_grad():   # centre at each age's mean log rate
                    self.head.bias.copy_(torch.as_tensor(bias_by_age,
                                                         dtype=torch.float32))

            def forward(self, x):                             # [B, L, A]
                z = torch.relu(self.conv(x[:, None, :, :]))
                return self.head(self.drop(z.flatten(1)))     # [B, A]

        return Net(self._logm_fill.mean(axis=1))

    def _targets(self, exclude_years):
        key = frozenset(exclude_years)
        cached = self._subset_cache.get(key)
        if cached is None:
            ts = [t for t in range(self.L, self.T) if t not in exclude_years]
            X = torch.stack([self._X[:, t - self.L:t].T for t in ts])   # [B, L, A]
            cached = (ts, X)
            self._subset_cache[key] = cached
        return cached

    def _loss(self, net, exclude_years):
        ts, X = self._targets(exclude_years)
        out = net(X)                                          # [B, A]
        D = self._D[:, ts].T
        E = self._E[:, ts].T
        w = self._w[:, ts].T
        return _poisson_deviance(out, D, E, w)

    def _val_loss(self, net, val_years):
        net.eval()
        with torch.no_grad():
            ts = sorted(val_years)
            X = torch.stack([self._X[:, t - self.L:t].T for t in ts])
            return _poisson_deviance(net(X), self._D[:, ts].T,
                                     self._E[:, ts].T, self._w[:, ts].T)

    def _point_logmx(self, h: int) -> np.ndarray:
        self.net_.eval()
        win = self._X[:, -self.L:].T.clone()                  # [L, A]
        out = np.empty((h, self.n_ages))
        with torch.no_grad():
            for step in range(h):
                pred = self.net_(win[None])[0]                # [A]
                out[step] = pred.cpu().numpy()
                win = torch.cat([win[1:], pred[None]], dim=0)
        return out

    def mc_sample_mx(self, h: int, n: int, rng: np.random.Generator) -> np.ndarray:
        g = _torch_seed_from(rng)
        torch.manual_seed(int(g.initial_seed()))
        self.net_.train()
        outs = np.empty((n, h, self.n_ages))
        with torch.no_grad():
            for i in range(n):
                win = self._X[:, -self.L:].T.clone()
                for step in range(h):
                    pred = self.net_(win[None])[0]
                    outs[i, step] = pred.cpu().numpy()
                    win = torch.cat([win[1:], pred[None]], dim=0)
        self.net_.eval()
        return np.exp(outs)

    def _insample_logmx(self):
        fitted = self._logm_fill.copy()
        with torch.no_grad():
            ts, X = self._targets(set())
            out = self.net_(X).cpu().numpy()                        # [B, A]
        for j, t in enumerate(ts):
            fitted[:, t] = out[j]
        return fitted


# ---------------------------------------------------------------------------
# 3. LSTMKt — LSTM on the Lee–Carter index
# ---------------------------------------------------------------------------

class LSTMKt(_TorchFamily):
    """Stage 1: the existing LeeCarterSVD (EM path included) supplies
    (alpha, beta, kappa). Stage 2: an LSTM forecasts kappa; innovation sigma
    from one-step training residuals. Forecast paths carry fresh Gaussian
    innovations per horizon (the family's own noise, exactly as RWD);
    parameter/seed uncertainty is the mechanism's job."""

    lr_grid = (1e-2, 3e-3)
    epochs_grid = (300, 800)

    def __init__(self, lr_grid=None, epochs_grid=None, seed: int = 20260825,
                 inner_val_years: int = 5, dropout: float = 0.05,
                 window: int = 10, hidden: int = 16):
        super().__init__(lr_grid, epochs_grid, seed, inner_val_years, dropout)
        self.Lw, self.hidden = int(window), int(hidden)

    def _prepare(self, D, E):
        self._lc = LeeCarterSVD().fit(D, E)
        k = self._lc.kappa.astype(float)
        self._k_mean, self._k_std = float(k.mean()), float(k.std(ddof=1) or 1.0)
        self._kn = (k - self._k_mean) / self._k_std
        if self.T <= self.Lw + self.inner_val_years + 2:
            raise ValueError(f"need > {self.Lw + self.inner_val_years + 2} "
                             f"years for LSTM window {self.Lw}; got {self.T}")

    def _build(self):
        H_, drop = self.hidden, self.dropout

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(1, H_, batch_first=True)
                self.drop = nn.Dropout(drop)
                self.head = nn.Linear(H_, 1)

            def forward(self, x):                             # [B, L, 1]
                z, _ = self.lstm(x)
                return self.head(self.drop(z[:, -1]))[:, 0]   # [B]

        return Net()

    def _pairs(self, exclude_years):
        key = frozenset(exclude_years)
        cached = self._subset_cache.get(key)
        if cached is None:
            ts = [t for t in range(self.Lw, self.T) if t not in exclude_years]
            X = torch.as_tensor(
                np.stack([self._kn[t - self.Lw:t] for t in ts])[:, :, None],
                dtype=torch.float32).to(self.device)
            y = torch.as_tensor(np.asarray([self._kn[t] for t in ts]),
                                dtype=torch.float32).to(self.device)
            cached = (X, y)
            self._subset_cache[key] = cached
        return cached

    def _loss(self, net, exclude_years):
        X, y = self._pairs(exclude_years)
        return ((net(X) - y) ** 2).mean()

    def _val_loss(self, net, val_years):
        net.eval()
        with torch.no_grad():
            X, y = self._pairs(set(range(self.T)) - set(val_years))
            return ((net(X) - y) ** 2).mean()

    def fit(self, D, E):
        super().fit(D, E)
        with torch.no_grad():
            X, y = self._pairs(set())
            resid = (self.net_(X) - y).cpu().numpy() * self._k_std
        self.sigma_ = float(np.std(resid, ddof=1))
        return self

    def _k_paths(self, h: int, n: int, rng: np.random.Generator,
                 dropout: bool) -> np.ndarray:
        """[n, h] kappa paths: recursive LSTM mean + N(0, sigma) innovations."""
        self.net_.train() if dropout else self.net_.eval()
        win = np.broadcast_to(self._kn[-self.Lw:], (n, self.Lw)).copy()
        eps = rng.normal(0.0, self.sigma_, size=(n, h))
        out = np.empty((n, h))
        with torch.no_grad():
            for step in range(h):
                x = torch.as_tensor(win[:, :, None], dtype=torch.float32).to(self.device)
                mu = self.net_(x).cpu().numpy() * self._k_std + self._k_mean
                k_next = mu + eps[:, step]
                out[:, step] = k_next
                win = np.concatenate(
                    [win[:, 1:], ((k_next - self._k_mean) / self._k_std)[:, None]],
                    axis=1)
        self.net_.eval()
        return out

    def sample_mx(self, h: int, n: int, rng: np.random.Generator) -> np.ndarray:
        """[n, h, ages] paths with the family's own innovation noise (see
        class docstring) — dropout off; not degenerate, but carries no
        parameter uncertainty, which is why native stays inadmissible."""
        k = self._k_paths(h, n, rng, dropout=False)
        lc = self._lc
        return np.exp(lc.alpha[None, None, :] + k[:, :, None] * lc.beta[None, None, :])

    def mc_sample_mx(self, h: int, n: int, rng: np.random.Generator) -> np.ndarray:
        g = _torch_seed_from(rng)
        torch.manual_seed(int(g.initial_seed()))
        k = self._k_paths(h, n, rng, dropout=True)
        lc = self._lc
        return np.exp(lc.alpha[None, None, :] + k[:, :, None] * lc.beta[None, None, :])

    def _point_logmx(self, h: int) -> np.ndarray:
        lc = self._lc
        self.net_.eval()
        win = self._kn[-self.Lw:].copy()[None]
        out = np.empty(h)
        with torch.no_grad():
            for step in range(h):
                x = torch.as_tensor(win[:, :, None], dtype=torch.float32).to(self.device)
                mu = self.net_(x).cpu().numpy()[0] * self._k_std + self._k_mean
                out[step] = mu
                win = np.concatenate(
                    [win[:, 1:], [[(mu - self._k_mean) / self._k_std]]], axis=1)
        return lc.alpha[None, :] + out[:, None] * lc.beta[None, :]

    def fitted_mx(self) -> np.ndarray:
        return self._lc.fitted_mx()

    @property
    def n_ages(self):
        return self._n_ages

    @n_ages.setter
    def n_ages(self, v):
        self._n_ages = v


# ---------------------------------------------------------------------------
# 4. NBHead — distributional negative-binomial head
# ---------------------------------------------------------------------------

class NBHead(_CellFamily):
    """NB2 head: outputs (log mu-rate, log r). Native sampling draws the
    GAMMA mixing rate so the runner's registered Poisson composition on top
    reproduces exactly the NB predictive count law (spec §4 — one code path,
    no double count)."""

    lr_grid = (1e-2, 3e-3)
    epochs_grid = (200, 500)
    n_out = 2

    def _nll(self, out, D, E, w):
        log_mu = out[:, 0] + torch.log(torch.clamp(E, min=1.0))
        log_r = torch.clamp(out[:, 1], -7.0, 14.0)
        r = torch.exp(log_r)
        mu = torch.exp(log_mu)
        ll = (torch.lgamma(D + r) - torch.lgamma(r) - torch.lgamma(D + 1.0)
              + r * (log_r - torch.log(r + mu))
              + D * (log_mu - torch.log(r + mu)))
        return -(w * ll).sum() / w.sum()

    def _future_params(self, h: int):
        self.net_.eval()
        with torch.no_grad():
            out = self._forward_years(self.net_, np.arange(self.T, self.T + h))
        log_rate = out[:, :, 0].cpu().numpy()                       # [h, ages], per-exposure
        r = np.exp(np.clip(out[:, :, 1].cpu().numpy(), -7.0, 14.0))
        return log_rate, r

    def _point_logmx(self, h: int) -> np.ndarray:
        return self._future_params(h)[0]

    def sample_mx(self, h: int, n: int, rng: np.random.Generator) -> np.ndarray:
        """[n, h, ages] Gamma mixing rates: lambda ~ Gamma(r, mu/r) per cell."""
        log_rate, r = self._future_params(h)
        mu = np.exp(log_rate)
        return rng.gamma(shape=r[None], scale=(mu / r)[None], size=(n, h, self.n_ages))

    def mc_sample_mx(self, h: int, n: int, rng: np.random.Generator) -> np.ndarray:
        """Dropout-active parameter draws, each carrying its own Gamma rate."""
        g = _torch_seed_from(rng)
        torch.manual_seed(int(g.initial_seed()))
        self.net_.train()
        out = np.empty((n, h, self.n_ages))
        with torch.no_grad():
            for i in range(n):
                o = self._forward_years(self.net_, np.arange(self.T, self.T + h))
                mu = np.exp(o[:, :, 0].cpu().numpy())
                r = np.exp(np.clip(o[:, :, 1].cpu().numpy(), -7.0, 14.0))
                out[i] = rng.gamma(shape=r, scale=mu / r)
        self.net_.eval()
        return out

    def _insample_logmx(self):
        out = self._forward_years(self.net_, np.arange(self.T))
        return out[:, :, 0].T.cpu().numpy()
