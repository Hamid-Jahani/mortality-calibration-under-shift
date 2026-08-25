"""Experiment runner: one code path from (panel, regime) to a metrics table.

This module wires the pre-registered design together without adding any new
modelling content:

* ``MODELS`` / ``MECHANISMS`` — registries of the CLASSICAL model families and
  the UQ mechanisms in scope for the CPU sweep. Neural families, the GP, deep
  ensembles and MC dropout are deliberately absent (GPU budget, separate
  work); ``docs/GRID.md`` remains the authority on the full crossed design.
* ``run_cell`` — fit one (model, mechanism) cell on a training matrix pair and
  score its predictive samples against the observed test window through the
  single evaluation path (``mortcal.eval`` + ``mortcal.lifetable``,
  methodology rule 4). Returns one flat dict of python scalars.
* ``run_regime`` — sweep a pre-registered :class:`mortcal.splits.Regime` (or a
  tuple of them, e.g. the STABLE expanding origins) over populations x sexes x
  models x mechanisms and write one parquet row per cell. Cells whose fit or
  scoring raises are recorded with the exception string in the ``error``
  column — a broken cell never kills the sweep.

Conformal cells and proper scores
---------------------------------
The conformal wrappers emit samples drawn UNIFORMLY inside their intervals
(see the module docstring of ``mortcal/uq/conformal.py``): they construct
intervals, not predictive distributions, so CRPS / log score / PIT computed
from those samples are placeholders, not distributional claims. Every row
therefore carries a boolean ``scores_secondary`` column — True for conformal
mechanisms — and proper scores from flagged rows must never be ranked against
distributional mechanisms (they go to a flagged appendix table only).

Real-data guard
---------------
``run_regime`` REFUSES any regime not named ``"synthetic"`` unless
``allow_real=True`` is passed explicitly: PREREGISTRATION.md validation
gate 2 (R/StMoMo oracle parity) is still OPEN, and no real-data result may be
produced before it closes (see "Validation gates").
"""
from __future__ import annotations

import zlib
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import stats

from .eval import (
    crps_sample,
    interval_coverage,
    joint_path_coverage,
    log_score_poisson,
    pit_values,
    winkler_score,
)
from .lifetable import annuity_factor, life_expectancy
from .models import CBD, LeeCarterSVD, PoissonLeeCarter, RenshawHaberman, SparseVAR
from .splits import Regime
from .uq import CopulaPathConformal, EnbPIMx, PoissonBootstrap, SplitConformalMx

_RATE_FLOOR = 1e-10   # same floor as mortcal.models.lc / mortcal.uq.conformal
_LAM_FLOOR = 1e-12    # Poisson mean floor: logpmf(., 0) is -inf otherwise

#: Classical model families in scope for this runner (CPU sweep).
MODELS: dict[str, type] = {
    "LC": LeeCarterSVD,
    "PLC": PoissonLeeCarter,
    "CBD": CBD,
    "RH": RenshawHaberman,
    "SVAR": SparseVAR,
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

    ``mech_kwargs`` are forwarded to the mechanism wrapper (e.g. ``B`` for the
    bootstrap, ``cal_years`` for split conformal); ignored for "native".
    """
    check_admissible(model_name, mechanism)
    cls = MODELS[model_name]
    kw = dict(mech_kwargs or {})
    if mechanism == "native":
        return cls()
    if mechanism == "pboot":
        if not hasattr(cls, "fitted_mx"):
            # Admissible per docs/GRID.md, but the wrapper contract
            # (mortcal/uq/bootstrap.py) needs the in-sample fitted surface.
            raise NotImplementedError(
                f"({model_name}, pboot) is admissible per docs/GRID.md but "
                f"{cls.__name__} does not implement fitted_mx(), which "
                "PoissonBootstrap requires; implement fitted_mx() first")
        return PoissonBootstrap(cls, **kw)
    if mechanism == "split_conf":
        return SplitConformalMx(cls, **kw)
    if mechanism == "enbpi":
        return EnbPIMx(cls, **kw)
    if mechanism == "copula_conf":
        return CopulaPathConformal(cls, **kw)
    raise AssertionError(f"unhandled mechanism {mechanism!r}")  # pragma: no cover


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
) -> dict[str, float | bool]:
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
        the log score. Defaults to ``obs_E`` (the registered convention:
        realised exposures are treated as known offsets, not forecast).
    mech_kwargs : forwarded to the mechanism wrapper constructor.

    Returns
    -------
    One flat dict of python scalars (plus the boolean ``scores_secondary``
    flag — True for conformal mechanisms, whose CRPS / log score / PIT are
    placeholders computed from uniform-in-interval samples).

    Notes
    -----
    * Point scores (RMSE/MAE on log m_x) use the pointwise MEDIAN of the log
      predictive samples — invariant under the log transform and identical to
      the centring convention of the conformal wrappers.
    * Derived actuarial quantities (H5 inputs) are computed per horizon-1
      sample through ``mortcal.lifetable`` — mean, 2.5% and 97.5% quantiles
      of e0, e65 and the annuity-due factor ä65 @2%, plus their observed
      counterparts, so downstream interval coverage of derived quantities
      needs only these columns. On truncated synthetic panels with fewer
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
    samples_mx = est.sample_mx(h, n_samples, rng)          # [n, h, n_ages]

    log_samples = np.log(np.clip(samples_mx, _RATE_FLOOR, None))
    truth = np.log(np.clip(obs_D / obs_E, _RATE_FLOOR, None))   # [h, n_ages]

    point = np.median(log_samples, axis=0)
    out: dict[str, float | bool] = {
        "rmse_logmx": float(np.sqrt(np.mean((point - truth) ** 2))),
        "mae_logmx": float(np.mean(np.abs(point - truth))),
        "crps_logmx": float(np.mean(crps_sample(log_samples, truth))),
    }

    # Poisson log score on observed deaths (rounded inside log_score_poisson —
    # the pre-registered rounding convention for Lexis-split fractional deaths).
    lam = np.clip(samples_mx * E_fut[None, :, :], _LAM_FLOOR, None)
    out["poisson_log_score"] = float(np.mean(log_score_poisson(lam, obs_D)))

    for level in (0.50, 0.80, 0.95):
        tag = f"{int(round(level * 100))}"
        covered, _width = interval_coverage(log_samples, truth, level)
        out[f"coverage_{tag}"] = float(np.mean(covered))
        out[f"winkler_{tag}"] = float(np.mean(winkler_score(log_samples, truth, level)))

    out["joint_path_coverage_95"] = float(
        joint_path_coverage(log_samples, truth, 0.95))

    pit = pit_values(log_samples, truth, rng=rng)
    out["pit_ks_stat"] = float(stats.kstest(pit.ravel(), "uniform").statistic)

    # --- derived actuarial quantities from the horizon-1 sample table (H5) ---
    x_old = min(65, n_ages - 1)         # clamp only fires on truncated synthetic panels
    mx1 = samples_mx[:, 0, :]                                   # [n, n_ages]
    obs_mx1 = np.clip(obs_D[0] / obs_E[0], _RATE_FLOOR, None)
    for key, fn in (
        ("e0", lambda m: life_expectancy(m, 0)),
        ("e65", lambda m: life_expectancy(m, x_old)),
        ("ann65", lambda m: annuity_factor(m, x0=x_old, i=0.02)),
    ):
        vals = np.asarray(fn(mx1), dtype=float)                 # [n]
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_q025"] = float(np.quantile(vals, 0.025))
        out[f"{key}_q975"] = float(np.quantile(vals, 0.975))
        out[f"{key}_obs"] = float(fn(obs_mx1))

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
    """
    regimes: list[Regime] = [regime] if isinstance(regime, Regime) else list(regime)
    models = list(models)
    mechanisms = list(mechanisms)

    # --- REAL-DATA GUARD -------------------------------------------------
    # PREREGISTRATION.md validation gate 2 (R/StMoMo oracle parity) is still
    # OPEN: Python LC / Poisson-LC parameters have not yet been verified
    # against the R oracle, so no real-data forecast may be produced. Only
    # regimes explicitly named "synthetic" run without allow_real=True;
    # passing allow_real is a deliberate, auditable act reserved for after
    # the gate closes.
    for r in regimes:
        if r.name != "synthetic" and not allow_real:
            raise RuntimeError(
                f"run_regime refused regime {r.name!r}: PREREGISTRATION.md "
                "validation gate 2 (oracle parity) is still open — pass "
                "allow_real=True only once the gate has closed")

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
    df.to_parquet(out_path, index=False)
    return df
