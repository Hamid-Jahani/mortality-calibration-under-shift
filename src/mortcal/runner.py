"""Experiment runner: one code path from (panel, regime) to a metrics table.

This module wires the pre-registered design together without adding any new
modelling content:

* ``MODELS`` / ``MECHANISMS`` — registries of the CLASSICAL model families and
  the UQ mechanisms in scope for the CPU sweep. Neural families, the GP, deep
  ensembles and MC dropout are deliberately absent (GPU budget, separate
  work); ``docs/GRID.md`` remains the authority on the full crossed design.
* ``MODEL_KWARGS`` — per-family constructor arguments applied under EVERY
  mechanism (currently only CBD's age restriction, ``age_min=55``), so the
  family is the same object whichever mechanism wraps it.
* ``run_cell`` — fit one (model, mechanism) cell on a training matrix pair and
  score its predictive samples against the observed test window through the
  single evaluation path (``mortcal.eval`` + ``mortcal.lifetable``,
  methodology rule 4). Returns one flat dict of python scalars (plus two
  JSON-encoded vectors, see below).
* ``run_regime`` — sweep a pre-registered :class:`mortcal.splits.Regime` (or a
  tuple of them, e.g. the STABLE expanding origins) over populations x sexes x
  models x mechanisms and write one parquet row per cell. Cells whose fit or
  scoring raises are recorded with the exception string in the ``error``
  column — a broken cell never kills the sweep.

Columns emitted per cell (the analysis stage consumes these; nothing is
refit downstream):

===========================  ==================================================
scalar means                 rmse_logmx, mae_logmx, crps_logmx, crps_counts,
                             poisson_log_score, coverage_{50,80,95},
                             winkler_{50,80,95}, joint_path_coverage_95,
                             pit_ks_stat, pit_ks_pvalue, n_cells,
                             n_ages_scored, n_zero_death_cells
calibration by age (H4)      coverage_{50,80,95}_band{0_24,25_64,65_99},
                             pit_ks_band{0_24,25_64,65_99},
                             cov95_by_age (JSON list, one entry per panel
                             age = mean 95% hit indicator over horizons;
                             null where the age is not scored)
Murphy decomposition         murphy_{reliability,resolution,uncertainty,brier}
                             on the 95% hit indicators;
                             murphy_pit_{reliability,resolution,uncertainty}
                             on the PIT values; pit_hist (JSON, 10 bins)
per horizon (DM loss series) crps_h{k}, logscore_h{k}, coverage95_h{k},
                             winkler95_h{k} for k = 1..h
derived (H5)                 {e0,e65,ann65}_{point,q025,q975,obs,error}
flags                        scores_secondary (bool)
===========================  ==================================================

Scoring target (PREREGISTRATION-ADDENDUM-2)
-------------------------------------------
Rate-scale scores evaluate the predictive law of the OBSERVED rate, not the
latent one. Predictive m_x paths are made Poisson-inclusive — D* ~ Poisson(E
m_x*) on the model's own paths — and both sides of every rate-scale score use
the half-count continuity correction log(max(D, 0.5) / E). Scoring latent
samples against observed crude rates counts observation noise as
miscalibration: a correctly-specified Poisson-LC measured 0.10 / 0.19 / 0.28
against nominal 0.50 / 0.80 / 0.95 before this was fixed, and mechanisms
fitted or calibrated on observed residuals (SVAR, every conformal arm)
absorbed that noise and ranked better for a reason unrelated to any shift.
``n_zero_death_cells`` reports how many test cells carried the correction.

Conformal cells and proper scores
---------------------------------
The conformal wrappers emit samples drawn UNIFORMLY inside their intervals
(see the module docstring of ``mortcal/uq/conformal.py``): they construct
intervals, not predictive distributions, so CRPS / log score / PIT computed
from those samples are placeholders, not distributional claims. Every row
therefore carries a boolean ``scores_secondary`` column — True for conformal
mechanisms — and proper scores from flagged rows must never be ranked against
distributional mechanisms (they go to a flagged appendix table only).

A conformal mechanism is scored at its CONSTRUCTION level only (95%); the
``coverage_{50,80}``, ``winkler_{50,80}`` and ``coverage_{50,80}_band*``
columns are NaN on those rows (addendum 2 §3). Reading 50% / 80% quantiles
out of uniform-in-interval samples would describe the uniform filler rather
than a calibrated forecast, and would flatter these arms.

Undefined ages and masking
--------------------------
A family may be undefined on part of the age range: CBD (M5) is fit on ages
>= 55 only (``MODEL_KWARGS``) and returns NaN samples below that. The runner
scores ONLY ages on which every predictive sample and every observed rate is
finite (a per-age mask); ``n_ages_scored`` / ``n_cells`` record how many ages
and (horizon, age) cells entered each metric, and ``cov95_by_age`` is null on
masked ages. Means are never silently taken over a mixture of defined and
undefined cells.

Real-data guard
---------------
``run_regime`` REFUSES any regime not named ``"synthetic"`` unless
``allow_real=True`` is passed explicitly. PREREGISTRATION.md validation gate
2 (R/StMoMo oracle parity) closed on 2026-08-25 (``scripts/check_parity.py``);
the flag is retained so that producing a real-data result stays a deliberate,
auditable act rather than a default.
"""
from __future__ import annotations

import functools
import json
import zlib
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import stats

from .eval import (
    crps_counts,
    log_crude_rate,
    crps_sample,
    interval_coverage,
    joint_path_coverage,
    log_score_poisson,
    murphy_decomposition,
    murphy_pit,
    pit_values,
    winkler_score,
)
from .lifetable import annuity_factor, life_expectancy
from .models import CBD, LeeCarterSVD, PoissonLeeCarter, RenshawHaberman, SparseVAR
from .splits import Regime
from .uq import CopulaPathConformal, EnbPIMx, PoissonBootstrap, SplitConformalMx

_RATE_FLOOR = 1e-10   # same floor as mortcal.models.lc / mortcal.uq.conformal
_LAM_FLOOR = 1e-12    # Poisson mean floor: logpmf(., 0) is -inf otherwise

#: Coverage levels scored (nominal central intervals).
LEVELS: tuple[float, ...] = (0.50, 0.80, 0.95)

#: Age bands for calibration-by-age (H4). Inclusive edges; the LAST band is
#: open-ended above so panels topping out beyond 99 still get a band. Same
#: partition as the Mondrian bands of the conformal wrappers
#: (``mortcal.uq.conformal.DEFAULT_AGE_BANDS``), so band-level coverage of a
#: conformal cell is read at the resolution the wrapper calibrated at.
AGE_BANDS: tuple[tuple[int, int], ...] = ((0, 24), (25, 64), (65, 99))

#: PIT histogram bins for ``murphy_pit`` / ``pit_hist``.
PIT_BINS = 10

#: Classical model families in scope for this runner (CPU sweep).
MODELS: dict[str, type] = {
    "LC": LeeCarterSVD,
    "PLC": PoissonLeeCarter,
    "CBD": CBD,
    "RH": RenshawHaberman,
    "SVAR": SparseVAR,
}

#: Per-family constructor kwargs, applied under EVERY mechanism (the native
#: fit, every bootstrap refit, every conformal member and centre refit).
#: CBD: logit q_x is near-linear in age only at higher ages — the M5 fit is
#: restricted to ages 55-99 (Cairns, Blake & Dowd 2006; PREREGISTRATION.md
#: age-cap sensitivity), with samples NaN below 55 (masked in scoring).
MODEL_KWARGS: dict[str, dict] = {
    "CBD": {"age_min": 55},
}

#: UQ mechanisms in scope. "native" uses the model's own predictive law;
#: the others wrap the model class. Deep ensemble and MC dropout are
#: inadmissible for every classical family (docs/GRID.md: deterministic fits
#: have no seed variance) and are therefore not registered here at all.
MECHANISMS: tuple[str, ...] = (
    "native", "pboot", "split_conf", "enbpi", "copula_conf",
)

#: Mechanisms whose samples are uniform draws inside a conformal interval —
#: proper scores (CRPS / log score / PIT) from these cells are SECONDARY
#: (see module docstring and mortcal/uq/conformal.py).
CONFORMAL_MECHANISMS: frozenset[str] = frozenset(
    {"split_conf", "enbpi", "copula_conf"})

#: Admissible (model, mechanism) pairs — transcribed from docs/GRID.md rows
#: Lee-Carter (SVD), Poisson-LC, CBD (M5), APC/RH (M2-A), sparse VAR: every
#: classical family admits all five mechanisms here (the grid's dashes for
#: classical families are deep ensemble / MC dropout, which this runner does
#: not register). Kept explicit so a future registry addition MUST also be
#: added here consciously, mirroring the grid.
ADMISSIBLE: frozenset[tuple[str, str]] = frozenset(
    (m, u) for m in ("LC", "PLC", "CBD", "RH", "SVAR") for u in MECHANISMS
)


class InadmissibleCellError(ValueError):
    """Raised for (model, mechanism) pairs outside the registered grid."""


def check_admissible(model_name: str, mechanism: str) -> None:
    """Validate a grid cell against the registries and docs/GRID.md."""
    if model_name not in MODELS:
        raise InadmissibleCellError(
            f"unknown model {model_name!r}; registered: {sorted(MODELS)}")
    if mechanism not in MECHANISMS:
        raise InadmissibleCellError(
            f"unknown mechanism {mechanism!r}; registered: {list(MECHANISMS)} "
            "(deep ensemble / MC dropout are inadmissible for classical "
            "families per docs/GRID.md and are not registered)")
    if (model_name, mechanism) not in ADMISSIBLE:
        raise InadmissibleCellError(
            f"cell ({model_name}, {mechanism}) is inadmissible per docs/GRID.md")


def build_estimator(model_name: str, mechanism: str,
                    mech_kwargs: Mapping | None = None):
    """Construct the (unfitted) estimator for one admissible grid cell.

    ``MODEL_KWARGS[model_name]`` reach the family under every mechanism: the
    native instance directly, ``PoissonBootstrap`` through its
    ``**model_kwargs`` (base fit AND every refit), the conformal wrappers
    through a ``functools.partial`` base factory (every member fit and the
    centre refit). ``mech_kwargs`` are forwarded to the mechanism wrapper
    (e.g. ``B`` for the bootstrap, ``cal_years`` for split conformal);
    ignored for "native". All five classical families implement
    ``fitted_mx()`` (``mortcal/models``), the hook the bootstrap needs, so
    ``pboot`` is constructible for every registered family.
    """
    check_admissible(model_name, mechanism)
    cls = MODELS[model_name]
    model_kw = dict(MODEL_KWARGS.get(model_name, {}))
    factory = functools.partial(cls, **model_kw) if model_kw else cls
    kw = dict(mech_kwargs or {})
    if mechanism == "native":
        return factory()
    if mechanism == "pboot":
        return PoissonBootstrap(cls, **model_kw, **kw)
    if mechanism == "split_conf":
        return SplitConformalMx(factory, **kw)
    if mechanism == "enbpi":
        return EnbPIMx(factory, **kw)
    if mechanism == "copula_conf":
        return CopulaPathConformal(factory, **kw)
    raise AssertionError(f"unhandled mechanism {mechanism!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# helpers for one grid cell
# ---------------------------------------------------------------------------

def _band_masks(n_ages: int) -> list[tuple[str, np.ndarray]]:
    """(column tag, boolean age mask) per AGE_BANDS entry; last band open above."""
    ages = np.arange(n_ages)
    out = []
    for i, (lo, hi) in enumerate(AGE_BANDS):
        top = i == len(AGE_BANDS) - 1
        mask = (ages >= lo) if top else ((ages >= lo) & (ages <= hi))
        out.append((f"band{lo}_{hi}", mask))
    return out


def _mean_or_nan(x: np.ndarray) -> float:
    """Mean of x, NaN when there is nothing to average (an empty age band)."""
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)) if x.size else float("nan")


def _ks_uniform(x: np.ndarray) -> float:
    """KS distance of x from Uniform(0, 1); NaN on an empty set."""
    x = np.asarray(x, dtype=float).ravel()
    return float(stats.kstest(x, "uniform").statistic) if x.size else float("nan")


def _ks_uniform_pvalue(x: np.ndarray) -> float:
    """Nominal KS p-value against Uniform(0, 1) — DESCRIPTIVE ONLY.

    Registered by addendum 2 §4. The KS null assumes independent draws; PIT
    values across ages and horizons within a population are strongly
    dependent (one shared kappa path, neighbouring ages nearly collinear), so
    this p-value is anti-conservative and is reported as a descriptive
    companion to the statistic, never as a test. Formal inference on
    calibration uses the population-clustered procedures in
    ``mortcal.inference``.
    """
    x = np.asarray(x, dtype=float).ravel()
    return float(stats.kstest(x, "uniform").pvalue) if x.size else float("nan")


def _json_list(values: np.ndarray) -> str:
    """JSON array with non-finite entries encoded as null (strict JSON has
    no NaN literal; ``json.loads`` gives back ``None`` there)."""
    return json.dumps([float(v) if np.isfinite(v) else None for v in values])


def _death_samples(est, samples_mx: np.ndarray, E_fut: np.ndarray,
                   age_ok: np.ndarray, h: int, n_samples: int,
                   rng: np.random.Generator) -> np.ndarray:
    """[n, h, n_scored] predictive DEATH COUNTS for ``crps_counts``.

    Families exposing ``sample_deaths`` (Poisson-LC, RH) draw D ~ Poisson(m E)
    on a fresh path set from their own predictive law. For everything else
    (SVD-LC, CBD, SVAR, every wrapper) the same construction is composed here
    on the paths already drawn: lam = sample_mx * E_future, D = Poisson(lam)
    — literally what ``PoissonLeeCarter.sample_deaths`` does, so the count
    scale is the Poisson predictive law for every cell. ``Generator.poisson``
    rejects NaN means, hence the age mask is applied before composing.
    """
    if hasattr(est, "sample_deaths") and bool(age_ok.all()):
        return np.asarray(est.sample_deaths(E_fut, h, n_samples, rng), dtype=float)
    lam = samples_mx[:, :, age_ok] * E_fut[None, :, age_ok]
    return rng.poisson(np.clip(lam, 0.0, None)).astype(float)


def _derived_quantities(samples_mx: np.ndarray, obs_D: np.ndarray,
                        obs_E: np.ndarray, first_ok: int, contiguous: bool,
                        out: dict) -> None:
    """H5 inputs from the horizon-1 sample table: e0, e65, ä65 @ 2%.

    Point functional = pointwise MEDIAN of the per-sample values (see
    ``run_cell``); q025 / q975 = 2.5% / 97.5% sample quantiles; obs = the
    same functional of the observed horizon-1 rates; error = point - obs.

    Masked ages: the life table is built on the scored block
    ``ages first_ok..top`` re-indexed from 0. e_x = T_x / l_x and
    ä_x = sum_t v^t l_{x+t} / l_x are RATIOS to l_x, and every l_{x+t}, L_{x+t}
    with t >= 0 is proportional to l_x, so both are invariant to whatever the
    table does below x — including the infant a_0 rule that
    ``mortcal.lifetable`` applies to the first row of the truncated table,
    provided that row lies strictly below x. Hence e65 / ä65 are exact from
    a table starting at 55 (CBD), while e0 is undefined (NaN) whenever age 0
    is not scored. A model whose scored ages do not form one block ending at
    the top age breaks the open-group closure: all derived sample statistics
    are NaN then (observed values are always defined on the full panel).
    """
    n_ages = samples_mx.shape[2]
    x_old = min(65, n_ages - 1)          # clamp only fires on truncated synthetic panels
    obs_full = np.clip(obs_D[0] / obs_E[0], _RATE_FLOOR, None)
    mx1 = samples_mx[:, 0, first_ok:]                                      # [n, n_scored]
    for key, age, fn in (
        ("e0", 0, lambda m, x: life_expectancy(m, x)),
        ("e65", x_old, lambda m, x: life_expectancy(m, x)),
        ("ann65", x_old, lambda m, x: annuity_factor(m, x0=x, i=0.02)),
    ):
        obs_val = float(fn(obs_full, age))
        idx = age - first_ok
        # idx >= 1 whenever the table is truncated: the truncated table's row 0
        # takes the infant a_0 rule, which must stay strictly below x.
        defined = contiguous and idx >= 0 and (first_ok == 0 or idx >= 1)
        if defined:
            vals = np.asarray(fn(mx1, idx), dtype=float)                   # [n]
            point = float(np.median(vals))
            q025, q975 = float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))
        else:
            point = q025 = q975 = float("nan")
        out[f"{key}_point"] = point
        out[f"{key}_q025"] = q025
        out[f"{key}_q975"] = q975
        out[f"{key}_obs"] = obs_val
        out[f"{key}_error"] = point - obs_val


# ---------------------------------------------------------------------------
# one grid cell
# ---------------------------------------------------------------------------

def run_cell(
    D: np.ndarray,
    E: np.ndarray,
    model_name: str,
    mechanism: str,
    h: int,
    n_samples: int,
    rng: np.random.Generator,
    obs_D: np.ndarray | None = None,
    obs_E: np.ndarray | None = None,
    E_future: np.ndarray | None = None,
    mech_kwargs: Mapping | None = None,
) -> dict[str, float | int | bool | str]:
    """Fit one grid cell and score it on the observed test window.

    Parameters
    ----------
    D, E : [n_ages, n_train_years]
        Training deaths and central exposures (the panel matrices).
    model_name, mechanism : registry keys; validated against docs/GRID.md.
    h : forecast horizon (number of test years scored).
    n_samples : predictive m_x paths drawn from the fitted cell.
    rng : generator for sampling (and the randomised PIT).
    obs_D, obs_E : [h, n_ages]
        OBSERVED deaths and exposures of the test window — the truth the cell
        is scored against (observed log rates; rounded-deaths Poisson log
        score per PREREGISTRATION.md "Metrics"). Required.
    E_future : [h, n_ages], optional
        Exposures used to convert predictive m_x into Poisson death means for
        the log score and the count-scale CRPS. Defaults to ``obs_E`` (the
        registered convention: realised exposures are treated as known
        offsets, not forecast).
    mech_kwargs : forwarded to the mechanism wrapper constructor.

    Returns
    -------
    One flat dict: python floats / ints, the boolean ``scores_secondary``
    flag (True for conformal mechanisms, whose CRPS / log score / PIT are
    placeholders computed from uniform-in-interval samples) and two JSON
    strings (``cov95_by_age``, ``pit_hist``). Column glossary in the module
    docstring.

    Notes
    -----
    * **Point functional — one convention for every point metric.** The
      point forecast of ANY scored quantity is the pointwise MEDIAN of its
      predictive samples: log m_x per (horizon, age) cell for RMSE / MAE,
      and the per-sample life-table functionals e0, e65, ä65 for the
      ``*_point`` / ``*_error`` columns. The median is invariant under
      monotone transforms (median of log m_x = log of median m_x), sits
      inside the reported [q025, q975] interval by construction, is the
      centring convention of the conformal wrappers, and is unaffected by
      the heavy right tail of exponentiated Gaussian paths that would drag
      a sample mean. No column uses the sample mean.
    * **Age mask.** Only ages where every sample and every observed rate is
      finite are scored (module docstring, "Undefined ages and masking");
      ``n_ages_scored`` and ``n_cells`` (= h x n_ages_scored) record the
      denominator of every mean. Band and by-age columns are NaN / null
      where nothing is scored.
    * **Proper scores.** CRPS on log m_x (per-cell ``crps_sample``); Poisson
      log score on observed deaths rounded half-up inside
      ``log_score_poisson`` (the pre-registered convention for Lexis-split
      fractional deaths); ``crps_counts`` is its rounding-free sensitivity
      companion on sampled death counts against UNROUNDED observed deaths
      (``_death_samples`` documents the count construction).
    * **Murphy decomposition.** ``murphy_*`` applies Murphy (1973) to the
      95% interval-hit indicators with the constant forecast probability
      0.95 (classical exact form, one bin): reliability = (0.95 - empirical
      coverage)^2, resolution = 0, uncertainty = c(1 - c). It is the
      coverage gap on the Brier scale — reported because pre-registered, and
      read alongside ``murphy_pit_*`` (Broecker 2009 divergence form on the
      10-bin PIT histogram with the uniform as reference), whose reliability
      is the chi-square distance from uniformity that a single level's
      coverage cannot see.
    * **Per-horizon series.** ``crps_h{k}``, ``logscore_h{k}``,
      ``coverage95_h{k}``, ``winkler95_h{k}`` (k = 1..h) are the loss series
      per (pop, sex, origin) that the Diebold–Mariano / wild-cluster
      bootstrap layer (``mortcal.inference``) differences; the scalar means
      are their averages over horizons.
    * **Derived actuarial quantities (H5)** come from the horizon-1 sample
      table through ``mortcal.lifetable``; see ``_derived_quantities`` for
      the masked-age treatment. On truncated synthetic panels with fewer
      than 66 ages the "65" quantities are computed at the panel's top age
      (documented clamp; real panels are always ages 0-99).
    """
    if obs_D is None or obs_E is None:
        raise ValueError("obs_D and obs_E ([h, n_ages]) are required — the "
                         "cell is scored against the observed test window")
    obs_D = np.asarray(obs_D, dtype=float)
    obs_E = np.asarray(obs_E, dtype=float)
    n_ages = D.shape[0]
    if obs_D.shape != (h, n_ages) or obs_E.shape != (h, n_ages):
        raise ValueError(
            f"obs_D/obs_E must be [h={h}, n_ages={n_ages}], got "
            f"{obs_D.shape} / {obs_E.shape}")
    E_fut = obs_E if E_future is None else np.asarray(E_future, dtype=float)

    est = build_estimator(model_name, mechanism, mech_kwargs)
    est.fit(np.asarray(D, dtype=float), np.asarray(E, dtype=float))
    samples_mx = np.asarray(est.sample_mx(h, n_samples, rng), dtype=float)
    if samples_mx.shape != (n_samples, h, n_ages):
        raise ValueError(f"sample_mx returned {samples_mx.shape}, expected "
                         f"{(n_samples, h, n_ages)}")

    # Observed rate scale: half-count continuity correction (addendum 2 §2).
    with np.errstate(divide="ignore", invalid="ignore"):
        truth_full = log_crude_rate(obs_D, obs_E)                         # [h, n_ages]

    # --- age mask: score only ages defined in every sample and observation ---
    age_ok = (np.isfinite(samples_mx).all(axis=(0, 1))
              & np.isfinite(truth_full).all(axis=0))                      # [n_ages]
    if not age_ok.any():
        raise ValueError("no scorable ages: every age has a non-finite sample "
                         "or observation")
    first_ok = int(np.argmax(age_ok))
    contiguous = bool(age_ok[first_ok:].all())
    n_ok = int(age_ok.sum())

    # --- rate-scale predictive samples are POISSON-INCLUSIVE (addendum 2 §1) ---
    # The scored quantity is the OBSERVED crude rate, which carries Poisson
    # sampling noise; latent m_x* samples do not. Composing D* ~ Poisson(E m_x*)
    # on the model's own paths and converting to log crude rates gives the
    # predictive law of the quantity actually observed. Without this a
    # correctly-specified model is scored as badly miscalibrated (measured:
    # 0.10 / 0.19 / 0.28 against nominal 0.50 / 0.80 / 0.95), and mechanisms
    # fitted or calibrated on observed residuals — SVAR, every conformal arm —
    # absorb the noise and look better for a reason unrelated to the shift.
    E_scored = E_fut[None, :, age_ok]
    lam_rate = np.clip(samples_mx[:, :, age_ok] * E_scored, 0.0, None)
    log_samples = log_crude_rate(rng.poisson(lam_rate).astype(float), E_scored)
    truth = truth_full[:, age_ok]                                         # [h, n_ok]
    bands = [(tag, mask[age_ok]) for tag, mask in _band_masks(n_ages)]

    out: dict[str, float | int | bool | str] = {
        "n_ages_scored": n_ok,
        "n_cells": int(h * n_ok),
        "n_zero_death_cells": int((obs_D[:, age_ok] == 0).sum()),
    }

    # --- point metrics (pointwise median, see Notes) and proper scores ---
    point = np.median(log_samples, axis=0)
    out["rmse_logmx"] = float(np.sqrt(np.mean((point - truth) ** 2)))
    out["mae_logmx"] = float(np.mean(np.abs(point - truth)))
    crps_cells = crps_sample(log_samples, truth)                          # [h, n_ok]
    out["crps_logmx"] = float(np.mean(crps_cells))

    lam = np.clip(samples_mx[:, :, age_ok] * E_fut[None, :, age_ok], _LAM_FLOOR, None)
    ls_cells = log_score_poisson(lam, obs_D[:, age_ok])                   # [h, n_ok]
    out["poisson_log_score"] = float(np.mean(ls_cells))

    d_samp = _death_samples(est, samples_mx, E_fut, age_ok, h, n_samples, rng)
    out["crps_counts"] = float(np.mean(crps_counts(d_samp, obs_D[:, age_ok])))

    # --- interval metrics: overall and by age band (H4) ---
    # A conformal mechanism constructs ONE interval, at its construction level
    # (95%). Reading 50% / 80% quantiles out of uniform-in-interval samples
    # would describe the uniform filler, not a calibrated forecast, and would
    # flatter these arms: a uniform on [lo, hi] puts exactly 50% of its mass in
    # the middle half, so coverage_50 would trend toward nominal by
    # construction regardless of whether the interval is any good. Addendum 2
    # §3 registers those columns as not-applicable for conformal cells.
    is_conformal = mechanism in CONFORMAL_MECHANISMS
    scored_levels = (0.95,) if is_conformal else LEVELS

    cov = {}
    wink95 = None
    for level in LEVELS:
        tag = f"{int(round(level * 100))}"
        if level not in scored_levels:
            out[f"coverage_{tag}"] = float("nan")
            out[f"winkler_{tag}"] = float("nan")
            for btag, _sel in bands:
                out[f"coverage_{tag}_{btag}"] = float("nan")
            continue
        covered, _width = interval_coverage(log_samples, truth, level)    # [h, n_ok]
        wink = winkler_score(log_samples, truth, level)
        cov[level] = covered.astype(float)
        out[f"coverage_{tag}"] = float(np.mean(covered))
        out[f"winkler_{tag}"] = float(np.mean(wink))
        for btag, sel in bands:
            out[f"coverage_{tag}_{btag}"] = _mean_or_nan(covered[:, sel])
        if level == 0.95:
            wink95 = wink
    cov95 = cov[0.95]

    out["joint_path_coverage_95"] = float(
        joint_path_coverage(log_samples, truth, 0.95))

    # --- PIT: overall, by band, and the Murphy decompositions ---
    pit = pit_values(log_samples, truth, rng=rng)                         # [h, n_ok]
    out["pit_ks_stat"] = _ks_uniform(pit)
    out["pit_ks_pvalue"] = _ks_uniform_pvalue(pit)                        # descriptive
    for btag, sel in bands:
        out[f"pit_ks_{btag}"] = _ks_uniform(pit[:, sel])

    md = murphy_decomposition(np.full(cov95.size, 0.95), cov95, n_bins=None)
    for k in ("reliability", "resolution", "uncertainty", "brier"):
        out[f"murphy_{k}"] = float(md[k])
    mp = murphy_pit(pit, n_bins=PIT_BINS)
    for k in ("reliability", "resolution", "uncertainty"):
        out[f"murphy_pit_{k}"] = float(mp[k])
    out["pit_hist"] = _json_list(np.asarray(mp["hist"], dtype=float))

    # --- per-horizon loss series (Diebold-Mariano inputs) ---
    for j in range(h):
        out[f"crps_h{j + 1}"] = float(np.mean(crps_cells[j]))
        out[f"logscore_h{j + 1}"] = float(np.mean(ls_cells[j]))
        out[f"coverage95_h{j + 1}"] = float(np.mean(cov95[j]))
        out[f"winkler95_h{j + 1}"] = float(np.mean(wink95[j]))

    # --- per-age 95% coverage curve (mean over horizons), null when masked ---
    curve = np.full(n_ages, np.nan)
    curve[age_ok] = cov95.mean(axis=0)
    out["cov95_by_age"] = _json_list(curve)

    # --- derived actuarial quantities from the horizon-1 sample table (H5) ---
    _derived_quantities(samples_mx, obs_D, obs_E, first_ok, contiguous, out)

    out["scores_secondary"] = mechanism in CONFORMAL_MECHANISMS
    return out


# ---------------------------------------------------------------------------
# one regime sweep
# ---------------------------------------------------------------------------

def _pivot_matrices(sub: pd.DataFrame, train_max_year: int,
                    test_years: tuple[int, ...]):
    """(D, E, obs_D, obs_E) matrices for one (pop, sex) from the tidy panel.

    D, E are [n_ages, n_train_years] (all years <= origin, expanding window);
    obs_D, obs_E are [h, n_ages]. Raises on ragged/missing cells so the
    caller can record the error instead of silently modelling holes.
    """
    piv_D = sub.pivot(index="age", columns="year", values="D").sort_index()
    piv_E = sub.pivot(index="age", columns="year", values="E").sort_index()
    ages = piv_D.index.to_numpy()
    if ages[0] != 0 or not np.array_equal(ages, np.arange(len(ages))):
        raise ValueError(f"ages not contiguous from 0: {ages.min()}..{ages.max()}")
    years = piv_D.columns.to_numpy()
    train_years = [int(y) for y in years if y <= train_max_year]
    if not train_years:
        raise ValueError(f"no training years <= {train_max_year} in panel")
    missing = [y for y in test_years if y not in set(int(v) for v in years)]
    if missing:
        raise ValueError(f"test years missing from panel: {missing}")
    D = piv_D[train_years].to_numpy(dtype=float)
    E = piv_E[train_years].to_numpy(dtype=float)
    obs_D = piv_D[list(test_years)].to_numpy(dtype=float).T
    obs_E = piv_E[list(test_years)].to_numpy(dtype=float).T
    for name, arr in (("D", D), ("E", E), ("obs_D", obs_D), ("obs_E", obs_E)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"non-finite cells in {name} after pivot")
    return D, E, obs_D, obs_E


def _cell_seed(base_seed: int, origin: int, pop: str, sex: str,
               model_name: str, mechanism: str) -> np.random.SeedSequence:
    """Deterministic per-cell seed (methodology rule 7: seeds recorded).

    The entropy is the global seed plus stable CRC32 hashes of the cell key,
    so any single cell can be re-run in isolation and reproduce its row.
    """
    parts = [base_seed, origin] + [
        zlib.crc32(s.encode()) for s in (pop, sex, model_name, mechanism)]
    return np.random.SeedSequence(parts)


def run_regime(
    panel_df: pd.DataFrame,
    regime: Regime | Iterable[Regime],
    models: Iterable[str],
    mechanisms: Iterable[str],
    n_samples: int,
    out_path,
    allow_real: bool = False,
    base_seed: int = 20260825,
    mech_kwargs: Mapping | None = None,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Sweep regime(s) over (pop, sex, origin, model, mechanism); write parquet.

    Parameters
    ----------
    panel_df : tidy panel with columns pop, year, age, sex, D, E
        (the shape produced by ``mortcal.data.hmd.build_panel``).
    regime : one :class:`~mortcal.splits.Regime` or an iterable of them
        (pass ``mortcal.splits.STABLE`` for the expanding-origin control —
        each origin is its own Regime).
    models, mechanisms : registry names; every pair is validated against
        docs/GRID.md before anything is fit.
    n_samples : predictive paths per cell.
    out_path : parquet output path (pandas + pyarrow), one row per
        (pop, sex, origin, model, mechanism).
    allow_real : see the guard below.
    base_seed : global seed; per-cell generators derive from it via
        :func:`_cell_seed`, recorded in the ``seed_entropy`` column.
    mech_kwargs : forwarded to every mechanism wrapper constructor.
    log : sink for skip messages (default ``print``).

    Failure policy: a cell whose matrix build, fit or scoring raises is
    SKIPPED and logged; its row carries the exception string in the ``error``
    column (metrics NaN) so the sweep always completes and the failure is
    auditable in the results table itself.

    Per-horizon columns are named for the regime's horizons (``crps_h1`` ..
    ``crps_h{H}``); when regimes with different H share one parquet file
    (STABLE origins capped at 2019) the missing horizons are NaN.
    """
    regimes: list[Regime] = [regime] if isinstance(regime, Regime) else list(regime)
    models = list(models)
    mechanisms = list(mechanisms)

    # --- REAL-DATA GUARD -------------------------------------------------
    # PREREGISTRATION.md validation gate 2 (R/StMoMo oracle parity) closed on
    # 2026-08-25 (scripts/check_parity.py, results/parity/). The guard stays:
    # only regimes explicitly named "synthetic" run without allow_real=True,
    # so producing a real-data table remains a deliberate, auditable act.
    for r in regimes:
        if r.name != "synthetic" and not allow_real:
            raise RuntimeError(
                f"run_regime refused regime {r.name!r}: real-data regimes "
                "require an explicit allow_real=True (PREREGISTRATION.md "
                "validation gate 2 closed 2026-08-25; the flag records that "
                "a real-data result was produced deliberately)")

    for m in models:
        for u in mechanisms:
            check_admissible(m, u)      # fail the whole sweep BEFORE any fit

    rows: list[dict] = []
    for r in regimes:
        origin = r.train_max_year
        h = len(r.test_years)
        for pop in r.pops:
            sub_pop = panel_df[panel_df["pop"] == pop]
            sexes = sorted(sub_pop["sex"].unique())
            if not sexes:
                sexes = ["<absent>"]
            for sex in sexes:
                sub = sub_pop[sub_pop["sex"] == sex]
                try:
                    mats = _pivot_matrices(sub, origin, r.test_years)
                    build_err = None
                except Exception as exc:  # noqa: BLE001 — recorded, not hidden
                    mats, build_err = None, f"{type(exc).__name__}: {exc}"
                for model_name in models:
                    for mechanism in mechanisms:
                        ss = _cell_seed(base_seed, origin, pop, sex,
                                        model_name, mechanism)
                        row = {
                            "regime": r.name, "pop": pop, "sex": sex,
                            "origin": origin, "model": model_name,
                            "mechanism": mechanism, "h": h,
                            "n_samples": n_samples,
                            "seed_entropy": str(list(ss.entropy)),
                            "error": None,
                        }
                        if build_err is not None:
                            row["error"] = build_err
                            log(f"SKIP {r.name}/{pop}/{sex}/{origin}/"
                                f"{model_name}/{mechanism}: {build_err}")
                            rows.append(row)
                            continue
                        D, E, obs_D, obs_E = mats
                        try:
                            row.update(run_cell(
                                D, E, model_name, mechanism, h, n_samples,
                                np.random.default_rng(ss),
                                obs_D=obs_D, obs_E=obs_E,
                                mech_kwargs=mech_kwargs))
                        except Exception as exc:  # noqa: BLE001
                            row["error"] = f"{type(exc).__name__}: {exc}"
                            log(f"SKIP {r.name}/{pop}/{sex}/{origin}/"
                                f"{model_name}/{mechanism}: {row['error']}")
                        rows.append(row)

    df = pd.DataFrame(rows)
    if "scores_secondary" in df.columns:
        # nullable boolean: error rows have no flag; parquet-safe via pyarrow
        df["scores_secondary"] = df["scores_secondary"].astype("boolean")
    for col in ("n_ages_scored", "n_cells"):
        if col in df.columns:
            df[col] = df[col].astype("Int64")     # nullable int: NaN on error rows
    df.to_parquet(out_path, index=False)
    return df
