#!/usr/bin/env python
"""Figure stage: sweep parquet -> paper/figures/*.pdf (vector, matplotlib only).

    python scripts/make_figures.py                       # results/<regime>.parquet (+ <regime>_gp.parquet)
    python scripts/make_figures.py --source shift=results/_shift_snapshot.parquet   # smoke on a snapshot
    python scripts/make_figures.py --regimes shift --cells all                     # uncensored appendix view

Consumes what the runner emitted (column contract: the docstring of
``src/mortcal/runner.py``); refits and re-scores nothing. Four figures per
registered regime (shift, placebo, stable), one file each:

==============================  ==================================================
fig-coverage-by-age-<regime>    H4: 95% coverage by single age (``cov95_by_age``),
                                one panel per family, mechanisms as lines
fig-pit-hist-<regime>           PIT histograms (``pit_hist``, 10 bins), family x
                                DISTRIBUTIONAL mechanism grid
fig-reliability-<regime>        H2: nominal (0.50/0.80/0.95) vs empirical coverage
                                per family/mechanism; conformal arms at 0.95 only
fig-joint-vs-marginal-<regime>  H3: paired bars ``coverage_95`` vs
                                ``joint_path_coverage_95`` per family/mechanism
==============================  ==================================================

Scoring discipline (runner docstring; PREREGISTRATION-ADDENDUM-2 s3; -3 s11):

* Conformal arms (``split_conf``, ``enbpi``, ``copula_conf``; rows flagged
  ``scores_secondary``) construct ONE interval at 95%: their CRPS / log score /
  PIT are placeholders and their 50% / 80% columns are NaN by design. They
  never enter the PIT figure, are read at 0.95 only in the reliability figure,
  and are drawn in a distinct style everywhere (dashed line, open marker,
  dashed bar edge, "(c)" suffix).
* CBD (M5) is scored on ages 55-99 (``n_ages_scored`` = 45). Every figure is
  panelled BY FAMILY, so CBD never averages with a full-age family; its age
  support is printed in its panel title.
* Error rows (``error`` not null) enter no mean; the footer of every figure
  states how many were excluded. The design-floor / method-failure split is
  ``scripts/final_qa.py``'s job, not this script's.
* Within a family panel the mechanisms are compared on the INTERSECTION of
  (origin, pop, sex) cells in which every mechanism present produced a valid
  row (addendum 3 s11, the same rule ``mortcal.inference.losses_from_rows``
  applies); the intersection size is printed in the panel title.
  ``--cells all`` gives the uncensored per-arm view for the appendix.
* An absent regime (no ``results/<regime>.parquet``) yields a clearly
  labelled PENDING placeholder under the SAME filename, so the LaTeX build
  never breaks and the reader sees what is missing. A source whose basename
  starts with ``_`` (a snapshot) or ``--provisional`` stamps every figure with
  "GENERATED SNAPSHOT - NOT FINAL" (banner + PDF Subject metadata) - the PDF
  analogue of the first-line comment on generated tables.

Palette: Okabe-Ito, fixed hue per mechanism (colour follows the entity, never
its rank); validated with the dataviz palette checker on 2026-08-28 - CVD
separation PASS (worst adjacent dE 9.6), normal-vision PASS; black is the
neutral reference arm (model-native) and carries line style + marker as
secondary encoding. A legend is always present.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mortcal.runner import (ADMISSIBLE, CONFORMAL_MECHANISMS,  # noqa: E402
                            MECHANISMS, MODELS, SECONDARY)

# ---------------------------------------------------------------------------
# registries and labels
# ---------------------------------------------------------------------------

REGIMES: tuple[str, ...] = ("shift", "placebo", "stable")
REGIME_TITLE = {
    "shift": "shift regime (train <= 2019, test 2020-2024)",
    "placebo": "placebo regime (train <= 1913, test 1914-1922)",
    "stable": "stable regime (expanding origins 1990-2014, pooled)",
}
FAMILY_ORDER: tuple[str, ...] = tuple(MODELS)      # LC PLC CBD RH SVAR GP NLC CNN LSTM NB
FAMILY_LABEL = {
    "LC": "Lee-Carter (SVD)", "PLC": "Poisson Lee-Carter", "CBD": "CBD (M5)",
    "RH": "APC / Renshaw-Haberman", "SVAR": "sparse VAR", "GP": "multi-output GP",
    "NLC": "neural-LC (R-W)", "CNN": "shallow CNN-LC", "LSTM": "LSTM on k_t",
    "NB": "distributional NB head",
}
MECH_ORDER: tuple[str, ...] = tuple(MECHANISMS)
CONFORMAL: frozenset[str] = frozenset(CONFORMAL_MECHANISMS)
DISTRIBUTIONAL: tuple[str, ...] = tuple(m for m in MECH_ORDER if m not in CONFORMAL)
MECH_LABEL = {
    "native": "model-native", "pboot": "Poisson bootstrap",
    "ensemble": "deep ensemble (M=10)", "dropout": "MC dropout",
    "split_conf": "split conformal", "enbpi": "EnbPI", "copula_conf": "copula conformal",
}
MECH_SHORT = {
    "native": "native", "pboot": "pboot", "ensemble": "ensemble", "dropout": "dropout",
    "split_conf": "split", "enbpi": "EnbPI", "copula_conf": "copula",
}
#: Okabe-Ito. Fixed assignment: a mechanism keeps its hue in every figure and
#: every regime, whichever arms happen to be present.
MECH_COLOR = {
    "native": "#000000", "pboot": "#E69F00", "ensemble": "#009E73",
    "dropout": "#0072B2", "split_conf": "#D55E00", "enbpi": "#CC79A7",
    "copula_conf": "#56B4E9",
}
MECH_MARKER = {
    "native": "o", "pboot": "s", "ensemble": "^", "dropout": "v",
    "split_conf": "D", "enbpi": "P", "copula_conf": "X",
}
LEVELS: tuple[float, ...] = (0.50, 0.80, 0.95)
NOMINAL = 0.95
FIGURES: tuple[str, ...] = ("fig-coverage-by-age", "fig-pit-hist",
                            "fig-reliability", "fig-joint-vs-marginal")
CELL_KEYS: tuple[str, ...] = ("origin", "pop", "sex")
GREY = "0.45"
REF_LS = (0, (3, 2))           # dashed reference lines (nominal, identity)
MANIFEST = "figures-manifest.json"

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 7, "axes.titlesize": 6.5, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "0.3", "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": "0.92", "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.1,
    "hatch.linewidth": 0.5,
})


# ---------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------

def default_sources(results_dir: Path, regime: str) -> list[Path]:
    """The regime's main parquet plus the GP pass (``launch_sweeps.sh``), whichever exist."""
    cands = [results_dir / f"{regime}.parquet", results_dir / f"{regime}_gp.parquet"]
    return [p for p in cands if p.exists()]


def load_regime(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        d = pd.read_parquet(p)
        d["_source"] = Path(p).name
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    if "error" not in df.columns:
        df["error"] = None
    if "scores_secondary" in df.columns:
        # the flag is the runner's; the mechanism registry is the fallback.
        # Disagreement means the row was produced by a different registry.
        ok = df[df["error"].isna()]
        flag = ok["scores_secondary"].fillna(False).astype(bool)
        reg = ok["mechanism"].isin(CONFORMAL)
        if (flag != reg).any():
            warnings.warn("scores_secondary disagrees with the conformal registry on "
                          f"{int((flag != reg).sum())} rows; styling follows the registry")
    return df


def is_snapshot(paths: list[Path]) -> bool:
    return any(p.name.startswith("_") for p in paths)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_cells(fam_rows: pd.DataFrame, mode: str = "common",
                mechanisms=None):
    """Valid rows of ONE family, restricted per addendum 3 s11.

    Returns ``(ok, arms, n_common, n_total, failed)``:

    * ``ok`` - error-free rows; in ``mode == "common"`` only the (origin,
      pop, sex) cells in which EVERY arm in ``arms`` produced a valid row.
    * ``arms`` - mechanisms with >= 1 valid row, in registry order, limited
      to ``mechanisms`` when given (so a figure that never shows conformal
      arms is not censored by them).
    * ``n_common`` - intersection size (``None`` in ``mode == "all"``).
    * ``n_total`` - distinct cells the sweep attempted for this family.
    * ``failed`` - mechanisms present only as error rows.
    """
    keys = [k for k in CELL_KEYS if k in fam_rows.columns]
    ok = fam_rows[fam_rows["error"].isna()]
    allowed = set(mechanisms) if mechanisms is not None else set(MECH_ORDER)
    arms = [m for m in MECH_ORDER if m in allowed and (ok["mechanism"] == m).any()]
    ok = ok[ok["mechanism"].isin(arms)]
    attempted = fam_rows[fam_rows["mechanism"].isin(allowed)]
    n_total = int(attempted[keys].drop_duplicates().shape[0]) if keys else int(len(attempted))
    failed = [m for m in MECH_ORDER if m in allowed
              and (attempted["mechanism"] == m).any() and m not in arms]
    n_common = None
    if mode == "common" and arms and keys:
        per_arm = [set(map(tuple, ok.loc[ok["mechanism"] == m, keys].itertuples(index=False)))
                   for m in arms]
        common = set.intersection(*per_arm)
        keep = [tuple(r) in common for r in ok[keys].itertuples(index=False)]
        ok = ok[np.asarray(keep, dtype=bool)] if len(ok) else ok
        n_common = len(common)
    return ok, arms, n_common, n_total, failed


def parse_json_col(series: pd.Series) -> np.ndarray:
    """[n_rows, n] float matrix from a JSON-list column; ``null`` -> NaN.

    Rows are padded with NaN to the longest list so a ragged column (a
    truncated synthetic panel) cannot raise.
    """
    rows = []
    for s in series:
        if s is None or (isinstance(s, float) and np.isnan(s)):
            continue
        v = json.loads(s) if isinstance(s, str) else list(s)
        rows.append(np.array([np.nan if x is None else float(x) for x in v]))
    if not rows:
        return np.empty((0, 0))
    n = max(len(r) for r in rows)
    out = np.full((len(rows), n), np.nan)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def _nanmean0(M: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(M, axis=0)


def _fmean(s: pd.Series) -> float:
    v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def _horizon(df: pd.DataFrame) -> int:
    if "h" in df.columns:
        h = pd.to_numeric(df["h"], errors="coerce").max()
        if np.isfinite(h):
            return int(h)
    ks = [int(c[len("coverage95_h"):]) for c in df.columns
          if c.startswith("coverage95_h") and c[len("coverage95_h"):].isdigit()]
    return max(ks) if ks else 1


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------

def _line_style(mech: str, markers: bool = True) -> dict:
    conf = mech in CONFORMAL
    c = MECH_COLOR[mech]
    st = dict(color=c, linestyle="--" if conf else "-")
    if markers:
        st.update(marker=MECH_MARKER[mech], markersize=3.6,
                  markerfacecolor="white" if conf else c, markeredgecolor=c,
                  markeredgewidth=0.8)
    return st


def _mech_handles(mechs) -> list:
    return [Line2D([0], [0], label=MECH_LABEL[m] + (" (c)" if m in CONFORMAL else ""),
                   **_line_style(m)) for m in mechs]


def _family_grid(title: str, sharex=True, sharey=True):
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 6.3), sharex=sharex, sharey=sharey)
    axes = axes.ravel()
    fam_ax = dict(zip(FAMILY_ORDER, axes[:len(FAMILY_ORDER)]))
    extra = list(axes[len(FAMILY_ORDER):])
    for ax in extra:
        ax.axis("off")
    fig.suptitle(title, fontsize=8.5, y=0.972)
    return fig, fam_ax, extra


def _empty(ax, text: str) -> None:
    ax.grid(False)
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=6, color="0.4", wrap=True)


def _panel_title(fam: str, n_common, n_total: int, support=None, per_arm=None) -> str:
    """Two lines: the family, then the cell count and (if restricted) age support."""
    parts = []
    if n_common is not None:
        parts.append(f"n={n_common}/{n_total} cells")
    elif per_arm:
        lo, hi = min(per_arm), max(per_arm)
        parts.append(f"n/arm={lo}" if lo == hi else f"n/arm={lo}-{hi}")
    if support is not None:
        parts.append(f"ages {support[0]}-{support[1]}")
    return FAMILY_LABEL[fam] + ("\n" + ", ".join(parts) if parts else "")


def _support_from(M: np.ndarray):
    """(first, last) age with a finite entry in a [rows, ages] matrix, else None."""
    if M.size == 0:
        return None
    fin = np.isfinite(M).any(axis=0)
    if not fin.any():
        return None
    idx = np.flatnonzero(fin)
    return int(idx[0]), int(idx[-1])


def _support_label(support, n_ages: int):
    """Print the support only when it is not the full panel 0..n_ages-1."""
    if support is None:
        return None
    if support == (0, n_ages - 1):
        return None
    return support


def stamp(fig, meta: dict) -> None:
    src = ", ".join(p.name for p in meta["sources"]) or "(none)"
    cells = ("common cells per family panel (addendum 3 s11)" if meta["cells"] == "common"
             else "all valid rows per arm (uncensored)")
    line1 = (f"source: {src} | rows {meta['n_rows']}: valid {meta['n_valid']}, "
             f"error rows {meta['n_err']} excluded from every mean | generated {meta['date']}")
    line2 = (f"{cells} | (c) = conformal arm: one interval at 95%, drawn dashed / "
             f"open marker / dashed bar edge; its proper scores are placeholders")
    fig.text(0.005, 0.016, line1, fontsize=5, color="0.35", ha="left", va="bottom")
    fig.text(0.005, 0.004, line2, fontsize=5, color="0.35", ha="left", va="bottom")
    if meta["snapshot"]:
        fig.text(0.5, 0.998, f"GENERATED SNAPSHOT - NOT FINAL - regenerate from "
                 f"results/{meta['regime']}.parquet", ha="center", va="top",
                 fontsize=7.5, color="#B00020", weight="bold")


def save(fig, path: Path, meta: dict, title: str) -> int:
    subject = (f"GENERATED SNAPSHOT - NOT FINAL - regenerate from results/{meta['regime']}.parquet"
               if meta["snapshot"] else
               f"generated from {', '.join(p.name for p in meta['sources']) or 'no source (pending)'}")
    fig.savefig(path, format="pdf",
                metadata={"Title": title, "Subject": subject,
                          "Creator": "scripts/make_figures.py"})
    plt.close(fig)
    return path.stat().st_size


def placeholder(meta: dict, fig_name: str, reason: str):
    fig = plt.figure(figsize=(7.2, 2.6))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.72, f"{fig_name} - {REGIME_TITLE[meta['regime']]}", ha="center",
            va="center", fontsize=9, weight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.45, f"PENDING: {reason}", ha="center", va="center", fontsize=8,
            color="#B00020", transform=ax.transAxes)
    ax.text(0.5, 0.25, "regenerate with  python scripts/make_figures.py  once the "
            f"regime parquet exists (results/{meta['regime']}.parquet)",
            ha="center", va="center", fontsize=6.5, color="0.35", transform=ax.transAxes)
    return fig


# ---------------------------------------------------------------------------
# the four figures
# ---------------------------------------------------------------------------

def fig_coverage_by_age(df: pd.DataFrame, meta: dict):
    title = f"95% interval coverage by single age - {REGIME_TITLE[meta['regime']]}"
    fig, fam_ax, extra = _family_grid(title)
    present: set[str] = set()
    for fam, ax in fam_ax.items():
        rows = df[df["model"] == fam]
        if rows.empty:
            ax.set_title(FAMILY_LABEL[fam])
            _empty(ax, "no rows in source(s)\n(pending)")
            continue
        ok, arms, n_common, n_total, failed = split_cells(rows, meta["cells"])
        if not arms or ok.empty or "cov95_by_age" not in ok.columns:
            ax.set_title(_panel_title(fam, n_common, n_total))
            _empty(ax, "no valid cells" + (f"\nfailed on every cell: {', '.join(failed)}" if failed else ""))
            continue
        support, n_ages, per_arm = None, 0, []
        for mech in arms:
            sub = ok[ok["mechanism"] == mech]
            M = parse_json_col(sub["cov95_by_age"])
            if M.size == 0:
                continue
            y = _nanmean0(M)
            x = np.arange(len(y))
            s = _support_from(M)
            if s is not None:
                support = s if support is None else (min(support[0], s[0]), max(support[1], s[1]))
            n_ages = max(n_ages, len(y))
            ax.plot(x, y, lw=1.0, **_line_style(mech, markers=False))
            present.add(mech)
            per_arm.append(len(sub))
        ax.axhline(NOMINAL, color=GREY, ls=REF_LS, lw=0.8, zorder=0)
        ax.set_title(_panel_title(fam, n_common, n_total,
                                  _support_label(support, n_ages), per_arm))
        if failed:
            ax.text(0.02, 0.04, "failed on every cell: " + ", ".join(failed),
                    transform=ax.transAxes, fontsize=5, color="0.35")
    for ax in fam_ax.values():
        ax.set_ylim(0, 1.03)
        ax.set_xlim(0, None)
    axes = list(fam_ax.values())
    for ax in axes[-4:]:
        ax.set_xlabel("age")
    for ax in axes[::4]:
        ax.set_ylabel("empirical 95% coverage")
    if extra:
        handles = _mech_handles([m for m in MECH_ORDER if m in present])
        handles.append(Line2D([0], [0], color=GREY, ls=REF_LS, lw=0.8, label="nominal 0.95"))
        extra[0].legend(handles=handles, loc="center", frameon=False, fontsize=6)
    if len(extra) > 1:
        extra[1].text(0.0, 0.5,
                      "Each line: mean over cells of the per-age 95%\n"
                      "hit rate (cov95_by_age; horizons pooled).\n"
                      "Ages a family does not score are absent\n"
                      "(CBD: 55-99). Solid = distributional arm,\n"
                      "dashed = conformal arm (interval at 0.95).",
                      fontsize=5.5, color="0.3", va="center", transform=extra[1].transAxes)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    stamp(fig, meta)
    return fig, title


def fig_pit_hist(df: pd.DataFrame, meta: dict):
    title = f"PIT histograms, distributional arms - {REGIME_TITLE[meta['regime']]}"
    nrow, ncol = len(FAMILY_ORDER), len(DISTRIBUTIONAL)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.6, 10.8), sharex=True, sharey=True)
    ymax = 0.3
    for i, fam in enumerate(FAMILY_ORDER):
        rows = df[df["model"] == fam]
        if rows.empty:
            ok, arms, n_common, n_total, failed = rows, [], None, 0, []
        else:
            ok, arms, n_common, n_total, failed = split_cells(
                rows, meta["cells"], mechanisms=DISTRIBUTIONAL)
        for j, mech in enumerate(DISTRIBUTIONAL):
            ax = axes[i, j]
            if (fam, mech) not in ADMISSIBLE:
                _empty(ax, "inadmissible\n(docs/GRID.md)")
                continue
            if rows.empty:
                _empty(ax, "pending")
                continue
            sub = ok[ok["mechanism"] == mech] if "pit_hist" in ok.columns else ok.iloc[0:0]
            if sub.empty:
                if mech in failed:
                    msg = "failed on every cell"
                elif mech not in arms:
                    msg = ("secondary cell (s),\nnot run" if (fam, mech) in SECONDARY
                           else "not run")
                else:
                    msg = "no common cells"
                _empty(ax, msg)
                continue
            H = parse_json_col(sub["pit_hist"])
            if H.size == 0:
                _empty(ax, "no pit_hist")
                continue
            tot = np.nansum(H, axis=1, keepdims=True)
            tot[tot == 0] = np.nan
            h = _nanmean0(H / tot)
            nb = len(h)
            centers = (np.arange(nb) + 0.5) / nb
            ax.bar(centers, h, width=0.9 / nb, color=MECH_COLOR[mech], linewidth=0)
            ax.axhline(1.0 / nb, color=GREY, ls=REF_LS, lw=0.8, zorder=3)
            ks = _fmean(sub["pit_ks_stat"]) if "pit_ks_stat" in sub.columns else float("nan")
            n_lab = f"n={len(sub)}" + ("" if n_common is None else f"/{n_total}")
            # top-centre: U-shaped PIT histograms (the common failure mode) are
            # lowest in the middle bins, so the label never sits on a tall bar
            ax.text(0.5, 0.94, n_lab + (f"\nKS={ks:.2f}" if np.isfinite(ks) else ""),
                    transform=ax.transAxes, ha="center", va="top", fontsize=5.2, color="0.25")
            if np.isfinite(h).any():
                ymax = max(ymax, float(np.nanmax(h)) * 1.08)
        axes[i, 0].set_ylabel(FAMILY_LABEL[fam] + ("\n(ages 55-99)" if fam == "CBD" else ""),
                              fontsize=6)
    for j, mech in enumerate(DISTRIBUTIONAL):
        axes[0, j].set_title(MECH_LABEL[mech], fontsize=7)
        axes[-1, j].set_xlabel("PIT")
    for ax in axes.ravel():
        ax.set_xlim(0, 1)
        ax.set_ylim(0, ymax)
        ax.set_xticks([0, 0.5, 1])
    fig.suptitle(title, fontsize=8.5, y=0.984)
    fig.text(0.5, 0.968, "bars: mean bin frequency over cells (each cell's histogram "
             "normalised to 1); dashed = uniform 1/10; KS = mean KS distance from U(0,1).\n"
             "Conformal arms are excluded: their PIT is a placeholder (addendum 2 s3).",
             ha="center", va="top", fontsize=5.5, color="0.3")
    fig.tight_layout(rect=(0, 0.03, 1, 0.945))
    stamp(fig, meta)
    return fig, title


def fig_reliability(df: pd.DataFrame, meta: dict):
    title = f"Reliability: nominal vs empirical central-interval coverage - {REGIME_TITLE[meta['regime']]}"
    fig, fam_ax, extra = _family_grid(title)
    present: set[str] = set()
    for fam, ax in fam_ax.items():
        rows = df[df["model"] == fam]
        if rows.empty:
            ax.set_title(FAMILY_LABEL[fam])
            _empty(ax, "no rows in source(s)\n(pending)")
            continue
        ok, arms, n_common, n_total, failed = split_cells(rows, meta["cells"])
        support = None
        if "cov95_by_age" in ok.columns and len(ok):
            M = parse_json_col(ok["cov95_by_age"])
            support = _support_label(_support_from(M), M.shape[1] if M.size else 0)
        ax.plot([0.4, 1.0], [0.4, 1.0], color=GREY, ls=REF_LS, lw=0.8, zorder=0)
        if not arms or ok.empty:
            ax.set_title(_panel_title(fam, n_common, n_total, support))
            _empty(ax, "no valid cells" + (f"\nfailed on every cell: {', '.join(failed)}" if failed else ""))
            continue
        conf_arms = [m for m in arms if m in CONFORMAL]
        dodge = dict(zip(conf_arms, np.linspace(-0.016, 0.016, len(conf_arms))
                         if len(conf_arms) > 1 else [0.0]))
        per_arm = []
        for mech in arms:
            sub = ok[ok["mechanism"] == mech]
            per_arm.append(len(sub))
            if mech in CONFORMAL:
                y = _fmean(sub["coverage_95"])
                if np.isfinite(y):
                    st = _line_style(mech)
                    st["linestyle"] = "none"
                    ax.plot([NOMINAL + dodge[mech]], [y], **st, zorder=4)
                    present.add(mech)
            else:
                ys = [_fmean(sub[f"coverage_{int(round(l * 100))}"]) for l in LEVELS]
                ax.plot(LEVELS, ys, lw=1.0, **_line_style(mech), zorder=3)
                present.add(mech)
        ax.set_title(_panel_title(fam, n_common, n_total, support, per_arm))
        if failed:
            ax.text(0.02, 0.94, "failed on every cell: " + ", ".join(failed),
                    transform=ax.transAxes, fontsize=5, color="0.35", va="top")
    for ax in fam_ax.values():
        ax.set_xlim(0.42, 1.0)
        ax.set_ylim(0, 1.03)
        ax.set_xticks(LEVELS)
        ax.set_xticklabels([f"{l:.2f}" for l in LEVELS])
    axes = list(fam_ax.values())
    for ax in axes[-4:]:
        ax.set_xlabel("nominal level")
    for ax in axes[::4]:
        ax.set_ylabel("empirical coverage")
    if extra:
        handles = _mech_handles([m for m in MECH_ORDER if m in present])
        handles.append(Line2D([0], [0], color=GREY, ls=REF_LS, lw=0.8, label="perfect reliability"))
        extra[0].legend(handles=handles, loc="center", frameon=False, fontsize=6)
    if len(extra) > 1:
        extra[1].text(0.0, 0.5,
                      "Points: mean coverage over cells at nominal\n"
                      "0.50 / 0.80 / 0.95 (central intervals).\n"
                      "Conformal arms construct one interval at 0.95\n"
                      "(50/80 columns NaN by design): open markers,\n"
                      "horizontally offset for legibility only.",
                      fontsize=5.5, color="0.3", va="center", transform=extra[1].transAxes)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    stamp(fig, meta)
    return fig, title


def fig_joint_vs_marginal(df: pd.DataFrame, meta: dict):
    H = _horizon(df)
    title = (f"Marginal vs joint path coverage at 95% (h = 1..{H}) - "
             f"{REGIME_TITLE[meta['regime']]}")
    fig, fam_ax, extra = _family_grid(title, sharex=False, sharey=True)
    w = 0.38
    for fam, ax in fam_ax.items():
        rows = df[df["model"] == fam]
        if rows.empty:
            ax.set_title(FAMILY_LABEL[fam])
            _empty(ax, "no rows in source(s)\n(pending)")
            ax.set_xticks([])
            continue
        ok, arms, n_common, n_total, failed = split_cells(rows, meta["cells"])
        support = None
        if "cov95_by_age" in ok.columns and len(ok):
            M = parse_json_col(ok["cov95_by_age"])
            support = _support_label(_support_from(M), M.shape[1] if M.size else 0)
        if not arms or ok.empty:
            ax.set_title(_panel_title(fam, n_common, n_total, support))
            _empty(ax, "no valid cells" + (f"\nfailed on every cell: {', '.join(failed)}" if failed else ""))
            ax.set_xticks([])
            continue
        per_arm, labels = [], []
        for k, mech in enumerate(arms):
            sub = ok[ok["mechanism"] == mech]
            per_arm.append(len(sub))
            c = MECH_COLOR[mech]
            conf = mech in CONFORMAL
            marg = _fmean(sub["coverage_95"])
            joint = _fmean(sub["joint_path_coverage_95"])
            edge = dict(edgecolor="black" if conf else c, linewidth=0.7,
                        linestyle="--" if conf else "-")
            if np.isfinite(marg):
                ax.bar(k - w / 2, marg, w, color=c, **edge, zorder=3)
            if np.isfinite(joint):
                ax.bar(k + w / 2, joint, w, color=c, alpha=0.45, hatch="////", **edge, zorder=3)
            labels.append(MECH_SHORT[mech] + (" (c)" if conf else ""))
        ax.axhline(NOMINAL, color=GREY, ls=REF_LS, lw=0.8, zorder=2)
        ax.axhline(NOMINAL ** H, color=GREY, ls=":", lw=0.8, zorder=2)
        ax.set_xticks(np.arange(len(arms)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=5.5)
        ax.set_xlim(-0.6, len(arms) - 0.4)
        ax.set_title(_panel_title(fam, n_common, n_total, support, per_arm))
        if failed:
            ax.text(0.02, 0.96, "failed on every cell: " + ", ".join(failed),
                    transform=ax.transAxes, fontsize=5, color="0.35", va="top")
    for ax in fam_ax.values():
        ax.set_ylim(0, 1.03)
    axes = list(fam_ax.values())
    for ax in axes[::4]:
        ax.set_ylabel("coverage")
    if extra:
        handles = [
            Patch(facecolor="0.55", edgecolor="0.55", label="marginal (coverage_95)"),
            Patch(facecolor="0.55", edgecolor="0.55", alpha=0.45, hatch="////",
                  label="joint path (all h jointly)"),
            Patch(facecolor="white", edgecolor="black", linestyle="--",
                  label="(c) conformal: interval bounds"),
            Line2D([0], [0], color=GREY, ls=REF_LS, lw=0.8, label="nominal 0.95"),
            Line2D([0], [0], color=GREY, ls=":", lw=0.8,
                   label=f"0.95^{H} = {NOMINAL ** H:.3f} (indep. horizons)"),
        ]
        extra[0].legend(handles=handles, loc="center left", bbox_to_anchor=(0.12, 0.5),
                        frameon=False, fontsize=5.5)
    if len(extra) > 1:
        extra[1].text(0.05, 0.5,
                      "Bar colour = mechanism (as in the\n"
                      "other figures). Joint path coverage:\n"
                      "share of (cell, age) paths inside the\n"
                      "95% band at EVERY horizon (H3).\n"
                      "Dotted line: joint rate of independent\n"
                      "0.95 marginals. The registered\n"
                      "comparator (model-implied joint\n"
                      "coverage, addendum 3 s8) is tabulated.",
                      fontsize=5.5, color="0.3", va="center", transform=extra[1].transAxes)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    stamp(fig, meta)
    return fig, title


BUILDERS = {
    "fig-coverage-by-age": fig_coverage_by_age,
    "fig-pit-hist": fig_pit_hist,
    "fig-reliability": fig_reliability,
    "fig-joint-vs-marginal": fig_joint_vs_marginal,
}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def make_regime(regime: str, paths: list[Path], outdir: Path, cells: str = "common",
                provisional: bool = False) -> dict:
    """Write the four PDFs for one regime; return the manifest entry."""
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {
        "regime": regime, "sources": list(paths), "cells": cells,
        "snapshot": provisional or is_snapshot(paths),
        "date": _dt.date.today().isoformat(),
        "n_rows": 0, "n_valid": 0, "n_err": 0,
    }
    entry = {
        "sources": [{"path": str(p), "sha256": _sha256(p)} for p in paths],
        "snapshot": meta["snapshot"], "cells": cells, "generated": meta["date"],
        "status": "pending" if not paths else "generated", "files": {},
    }
    df = None
    if paths:
        df = load_regime(paths)
        err = df["error"].notna()
        meta.update(n_rows=int(len(df)), n_valid=int((~err).sum()), n_err=int(err.sum()))
        entry.update(n_rows=meta["n_rows"], n_valid=meta["n_valid"], n_error_rows=meta["n_err"],
                     regime_labels=sorted(df["regime"].astype(str).unique().tolist())
                     if "regime" in df.columns else [])
    for name in FIGURES:
        path = outdir / f"{name}-{regime}.pdf"
        if df is None:
            reason = f"results/{regime}.parquet not present (regime not yet run)"
            fig = placeholder(meta, name, reason)
            stamp(fig, meta)
            title = f"{name} - {regime} - PENDING"
        else:
            fig, title = BUILDERS[name](df, meta)
        entry["files"][path.name] = save(fig, path, meta, title)
    return entry


def _parse_sources(items: list[str]) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--source expects REGIME=PATH[,PATH...], got {it!r}")
        regime, rest = it.split("=", 1)
        if regime not in REGIMES:
            raise SystemExit(f"unknown regime {regime!r}; registered: {REGIMES}")
        paths = [Path(p) for p in rest.split(",") if p]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"--source path(s) not found: {missing}")
        out[regime] = paths
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument("--out", default=str(ROOT / "paper" / "figures"))
    p.add_argument("--regimes", nargs="+", default=list(REGIMES), choices=REGIMES)
    p.add_argument("--source", action="append", default=[], metavar="REGIME=PATH[,PATH]",
                   help="override the regime's source parquet(s); a basename starting "
                        "with '_' is a snapshot and stamps NOT FINAL on every figure")
    p.add_argument("--cells", choices=("common", "all"), default="common",
                   help="common: intersection of cells valid for every arm in the panel "
                        "(addendum 3 s11, default); all: uncensored per-arm means")
    p.add_argument("--provisional", action="store_true",
                   help="stamp NOT FINAL regardless of the source name")
    args = p.parse_args(argv)

    results_dir, outdir = Path(args.results_dir), Path(args.out)
    overrides = _parse_sources(args.source)
    manifest_path = outdir / MANIFEST
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    for regime in args.regimes:
        paths = overrides.get(regime) or default_sources(results_dir, regime)
        entry = make_regime(regime, paths, outdir, args.cells, args.provisional)
        manifest[regime] = entry
        src = ", ".join(str(p) for p in paths) or "(none: PENDING placeholders)"
        flag = "  [SNAPSHOT - NOT FINAL]" if entry["snapshot"] else ""
        print(f"[figures] {regime}: source {src}{flag}", flush=True)
        if entry["status"] == "generated":
            print(f"[figures]   rows {entry['n_rows']}, valid {entry['n_valid']}, "
                  f"error rows {entry['n_error_rows']} excluded; cells={args.cells}", flush=True)
        for fname, size in entry["files"].items():
            print(f"[figures]   {fname:40s} {size:>9,d} bytes", flush=True)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"[figures] manifest -> {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
