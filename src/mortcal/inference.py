"""Forecast-comparison inference with population-level clustering.

The shift regime has effective sample size closer to ONE common shock than to
20 populations x 100 ages x 5 years (docs/IDEA.md, critic gate). Every test
here therefore treats the POPULATION as the cluster and resamples clusters,
never cells. Two procedures, both pre-registered (PREREGISTRATION.md,
"Metrics: Inference"):

* :func:`dm_wild_cluster` -- Diebold & Mariano (1995) comparison of two
  forecasters on a loss-differential series, with a wild cluster bootstrap
  (Cameron, Gelbach & Miller 2008) using Webb (2014/2023) six-point weights,
  which remain reliable at ~20 clusters where Rademacher weights offer too
  few distinct sign patterns for a smooth null distribution.
* :func:`model_confidence_set` -- Hansen, Lunde & Nason (2011) MCS with the
  T_max statistic and the e_max elimination rule; the bootstrap distribution
  is a CLUSTER bootstrap over populations (block = population), the analogue
  of their stationary block bootstrap for our panel structure.

Both functions take losses already aggregated to (cluster, unit) level; the
runner's per-horizon CRPS columns provide the unit dimension.
"""
from __future__ import annotations

import numpy as np

_WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def _cluster_t(d: np.ndarray, groups: np.ndarray) -> tuple[float, float, float]:
    """(mean, cluster-robust se, t) of differential d with integer cluster labels."""
    G = int(groups.max()) + 1
    mean = float(d.mean())
    resid = d - mean
    cluster_sums = np.bincount(groups, weights=resid, minlength=G)
    n = d.size
    var = (G / (G - 1)) * float((cluster_sums ** 2).sum()) / n ** 2   # CR1
    se = float(np.sqrt(var))
    return mean, se, (mean / se if se > 0 else np.inf)


def dm_wild_cluster(loss_a: np.ndarray, loss_b: np.ndarray, groups: np.ndarray,
                    n_boot: int = 4999, rng: np.random.Generator | None = None) -> dict:
    """Diebold-Mariano test of E[loss_a - loss_b] = 0 with a wild cluster bootstrap.

    loss_a, loss_b : per-unit losses of the two forecasters (same units)
    groups         : cluster label per unit (population code or int)
    Returns dict(mean_diff, se, t, p_value, n_clusters). Negative mean_diff
    favours forecaster a (losses are negatively oriented).

    The null is IMPOSED before resampling (residuals centred at the grand
    mean), per Cameron-Gelbach-Miller for size control with few clusters.
    """
    rng = rng or np.random.default_rng(0)
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    labels, inv = np.unique(np.asarray(groups), return_inverse=True)
    G = labels.size
    if G < 2:
        raise ValueError("need at least two clusters")
    mean, se, t = _cluster_t(d, inv)
    resid = d - mean                                   # null imposed
    t_star = np.empty(n_boot)
    for b in range(n_boot):
        w = rng.choice(_WEBB, size=G)
        d_b = resid * w[inv]                           # one weight per cluster
        m_b, se_b, _ = _cluster_t(d_b, inv)
        t_star[b] = m_b / se_b if se_b > 0 else 0.0
    p = float((np.abs(t_star) >= abs(t)).mean())
    return {"mean_diff": mean, "se": se, "t": float(t), "p_value": p, "n_clusters": int(G)}


def model_confidence_set(losses: np.ndarray, groups: np.ndarray, alpha: float = 0.10,
                         n_boot: int = 2000, rng: np.random.Generator | None = None,
                         names: list[str] | None = None) -> dict:
    """Hansen-Lunde-Nason (2011) Model Confidence Set with a cluster bootstrap.

    losses : [n_units, n_models] per-unit losses
    groups : cluster label per unit
    Returns dict(in_set, eliminated [(name, p)] in order, p_values {name: p}).

    Statistic T_max = max_i t_i with t_i = dbar_i / se(dbar_i), where
    dbar_i = mean_j (Lbar_i - Lbar_j). Elimination removes argmax_i t_i.
    Bootstrap: resample POPULATIONS with replacement (block = cluster) and
    recompute the centred statistic. The MCS p-value of the k-th eliminated
    model is the running maximum of the sequential p-values.
    """
    rng = rng or np.random.default_rng(0)
    L = np.asarray(losses, float)
    n, M = L.shape
    names = list(names) if names is not None else [f"m{i}" for i in range(M)]
    labels, inv = np.unique(np.asarray(groups), return_inverse=True)
    G = labels.size
    cluster_idx = [np.where(inv == g)[0] for g in range(G)]

    def centred_means(Lm: np.ndarray, rows: np.ndarray | None) -> np.ndarray:
        X = Lm if rows is None else Lm[rows]
        m = X.mean(axis=0)
        return m - m.mean()                            # dbar_i

    active = list(range(M))
    eliminated: list[tuple[str, float]] = []
    pvals: dict[str, float] = {}
    p_running = 0.0
    while len(active) > 1:
        Lm = L[:, active]
        d_i = centred_means(Lm, None)
        boot = np.empty((n_boot, len(active)))
        for b in range(n_boot):
            pick = rng.integers(0, G, size=G)
            rows = np.concatenate([cluster_idx[g] for g in pick])
            boot[b] = centred_means(Lm, rows) - d_i
        se = boot.std(axis=0, ddof=1)
        se = np.where(se > 0, se, np.inf)
        t = d_i / se
        t_max = float(t.max())
        t_max_star = (boot / se).max(axis=1)
        p = float((t_max_star >= t_max).mean())
        p_running = max(p_running, p)
        worst = active[int(np.argmax(t))]
        pvals[names[worst]] = p_running
        if p_running >= alpha:
            break
        eliminated.append((names[worst], p_running))
        active.remove(worst)
    for i in active:
        pvals.setdefault(names[i], 1.0)
    return {"in_set": [names[i] for i in active], "eliminated": eliminated, "p_values": pvals}
