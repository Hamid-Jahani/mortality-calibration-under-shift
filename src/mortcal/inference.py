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

Both functions take losses already aggregated to (cluster, unit) level;
:func:`losses_from_rows` builds exactly that from the runner's parquet rows,
applying the addendum 3 §11 common-cell restriction.

Non-finite losses RAISE. They are never imputed, dropped in place, or passed
through: on the runner's real NaN pattern (an arm failing on part of one
population) the MCS silently INVERTED — measured, 4 models x 20 clusters,
clean gave in_set ['A','B','C'] and one hole gave in_set ['BAD'] with the
three good models eliminated at p = 0.000. A wrong answer that looks right is
worse than a stopped analysis, and the correct handling of a missing cell is
the registered common-cell restriction, applied before any statistic.
"""
from __future__ import annotations

import numpy as np

_WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def _require_finite(L: np.ndarray, names: list[str] | None, what: str) -> None:
    """Raise unless every loss is finite (see module docstring)."""
    bad = ~np.isfinite(L)
    if not bad.any():
        return
    if L.ndim == 2:
        cols = np.flatnonzero(bad.any(axis=0))
        labels = [names[c] if names and c < len(names) else f"col{c}" for c in cols]
        detail = f" in {', '.join(labels)}"
    else:
        detail = ""
    raise ValueError(
        f"{what}: {int(bad.sum())} non-finite loss value(s){detail}. Losses are "
        "never imputed here — restrict to common cells first "
        "(mortcal.inference.losses_from_rows, addendum 3 §11).")


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
    a = np.asarray(loss_a, float)
    b = np.asarray(loss_b, float)
    _require_finite(a, None, "dm_wild_cluster(loss_a)")
    _require_finite(b, None, "dm_wild_cluster(loss_b)")
    d = a - b
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
    _require_finite(L, names, "model_confidence_set(losses)")
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


# ---------------------------------------------------------------------------
# runner rows -> (losses, groups) for the two procedures above
# ---------------------------------------------------------------------------

#: The cell key a runner row is scored on. A "unit" of the loss matrix is one
#: of these plus a horizon; a "cluster" is the population.
CELL_KEYS = ("regime", "pop", "sex", "origin")


def losses_from_rows(df, loss: str = "crps", arms=None, cluster: str = "pop",
                     allow_ragged_age_support: bool = False):
    """Build (losses, groups, names, report) from runner parquet rows.

    Parameters
    ----------
    df : the runner's tidy output — one row per
        (regime, pop, sex, origin, model, mechanism), carrying per-horizon
        loss columns ``{loss}_h1 .. {loss}_hH`` and an ``error`` column.
    loss : column prefix. ``"crps"`` (default) or ``"logscore"`` — the two
        per-horizon series the runner emits for Diebold-Mariano input.
    arms : optional list of (model, mechanism) pairs to compare. Restricting
        the arms also restricts what can censor a cell, which is how a
        contrast inside one sub-grid stays uncensored by an arm outside it
        (docs/GRID.md claims discipline).
    cluster : row column supplying the cluster label (default ``"pop"``).
    allow_ragged_age_support : permit arms scored over DIFFERENT numbers of
        ages. Off by default and rarely right — see below.

    Returns
    -------
    losses : [n_units, n_arms] float, guaranteed finite
    groups : [n_units] cluster labels
    names  : arm labels, ``"model/mechanism"``, in the column order of losses
    report : dict with ``n_units``, ``n_arms``, ``n_cells_kept``,
        ``n_cells_dropped``, ``dropped_cells``, ``arms_with_failures``,
        ``horizons`` — the intersection accounting addendum 3 §11 requires be
        reported alongside any contrast.

    **Common age support.** A family may be undefined on part of the age
    range — CBD (M5) is fit on ages 55+ — and the runner's per-horizon loss
    columns are already MEANS over whatever ages that family scored. A mean
    CRPS over ages 55-99 is not comparable with a mean over 0-99: mortality
    rates and their forecast errors vary by orders of magnitude across the
    age range. Observed on a synthetic sweep, an unguarded comparison put
    ``CBD/native`` alone in the model confidence set purely because it
    averaged over 45 ages while LC and PLC averaged over 100. Arms are
    therefore required to share ``n_ages_scored``; the honest ways to
    compare a restricted family are to run the others under the same
    restriction, or to report it in a separate sub-grid.
    ``allow_ragged_age_support=True`` overrides, and the resulting report
    carries ``ragged_age_support: True`` so the caveat travels with the
    numbers.

    **Addendum 3 §11.** A cell enters only if EVERY compared arm produced a
    valid row there. Arms fail on different cells (conformal arms need long
    panels, native arms die where a fit diverges), so an unrestricted
    comparison would contrast arms measured on different populations — a
    selection confound in exactly the direction of the headline claim. Cells
    are dropped whole, never per-arm, and the count is reported rather than
    left implicit in a denominator.
    """
    import pandas as pd  # local: keeps the module importable without pandas

    hcols = [c for c in df.columns
             if c.startswith(f"{loss}_h") and c[len(loss) + 2:].isdigit()]
    if not hcols:
        raise ValueError(
            f"no {loss}_h* columns in these rows; the runner emits "
            "crps_h*/logscore_h* per horizon")
    hcols = sorted(hcols, key=lambda c: int(c[len(loss) + 2:]))

    d = df.copy()
    d["_arm"] = d["model"].astype(str) + "/" + d["mechanism"].astype(str)
    if arms is not None:
        want = {f"{m}/{u}" for (m, u) in arms}
        d = d[d["_arm"].isin(want)]
        missing = want - set(d["_arm"])
        if missing:
            raise ValueError(f"requested arms absent from these rows: {sorted(missing)}")
    names = sorted(d["_arm"].unique())

    # Proper scores of conformal cells are PLACEHOLDERS (uniform-in-interval
    # samples; runner docstring, addendum 2 §3) and carry scores_secondary =
    # True. Ranking them - against distributional arms OR against each other -
    # ranks interval widths dressed as CRPS. Found on the first real snapshot
    # (2026-08-28): every conformal-family MCS was being decided on crps.
    # Interval-valid losses for those contrasts are the per-horizon Winkler
    # (winkler95_h*) or coverage (coverage95_h*) series.
    if loss in ("crps", "logscore") and "scores_secondary" in d.columns:
        flagged = sorted(set(d.loc[d["scores_secondary"].fillna(False).astype(bool), "_arm"]) & set(names))
        if flagged:
            raise ValueError(
                f"loss={loss!r} is a flagged secondary (placeholder) score for arm(s) "
                f"{flagged}; compare interval mechanisms on loss='winkler95' "
                "(or 'coverage95'), never on crps/logscore.")

    # a row is admissible only if it carries no error AND every horizon is finite
    ok = d["error"].isna() if "error" in d else pd.Series(True, index=d.index)
    finite = np.isfinite(d[hcols].to_numpy(dtype=float)).all(axis=1)
    d["_ok"] = ok.to_numpy() & finite

    keys = [k for k in CELL_KEYS if k in d.columns]
    good = d[d["_ok"]].groupby(keys, sort=True)["_arm"].nunique()
    complete = good[good == len(names)].index
    all_cells = d.groupby(keys, sort=True).size().index
    dropped = [c for c in all_cells if c not in set(complete)]
    if len(complete) == 0:
        raise ValueError(
            f"no cells have all {len(names)} arms: nothing to compare "
            "(addendum 3 §11 leaves an empty intersection)")

    keep = d.set_index(keys).loc[complete].reset_index()
    keep = keep[keep["_arm"].isin(names)]

    # one unit per (cell, horizon); arms become columns
    long = keep.melt(id_vars=keys + ["_arm"], value_vars=hcols,
                     var_name="_h", value_name="_loss")
    wide = long.pivot_table(index=keys + ["_h"], columns="_arm",
                            values="_loss", sort=True)
    wide = wide[names]
    losses = wide.to_numpy(dtype=float)
    _require_finite(losses, names, "losses_from_rows")
    groups = wide.index.get_level_values(cluster).to_numpy()

    support = None
    ragged = False
    if "n_ages_scored" in keep.columns:
        support = {a: sorted({int(v) for v in g})
                   for a, g in keep.groupby("_arm")["n_ages_scored"]}
        distinct = {n for v in support.values() for n in v}
        ragged = len(distinct) > 1
        if ragged and not allow_ragged_age_support:
            raise ValueError(
                "arms do not share an age support: "
                + ", ".join(f"{a}={v}" for a, v in sorted(support.items()))
                + ". The per-horizon loss columns are means over each family's "
                "OWN scored ages, so these are not comparable. Restrict the "
                "arms, re-run the others under the same age restriction, or "
                "pass allow_ragged_age_support=True to accept the caveat.")

    failing = sorted(set(d.loc[~d["_ok"], "_arm"]) & set(names))
    report = {
        "n_units": int(losses.shape[0]),
        "n_arms": len(names),
        "n_cells_kept": int(len(complete)),
        "n_cells_dropped": len(dropped),
        "dropped_cells": [tuple(c) if isinstance(c, tuple) else (c,) for c in dropped],
        "arms_with_failures": failing,
        "horizons": [int(c[len(loss) + 2:]) for c in hcols],
        "loss": loss,
        "age_support": support,
        "ragged_age_support": ragged,
    }
    return losses, groups, names, report
