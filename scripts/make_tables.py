"""Paper tables: runner parquet(s) + analysis JSON(s) -> paper/tables/*.tex.

    python scripts/make_tables.py --parquet results/shift.parquet \
        [--parquet results/shift_gp.parquet] [--parquet results/placebo.parquet ...] \
        --analysis results/shift_analysis.json [--analysis ...] \
        [--sensitivities results/sensitivities.json] \
        [--hmd-deaths Dataset/deaths/Deaths_1x1/Deaths_1x1.txt] \
        [--hmd-exposures Dataset/exposures/Exposures_1x1/Exposures_1x1.txt] \
        --out paper/tables

Consumes what ``scripts/final_qa.py`` has already gated and what
``scripts/analyse.py`` has already tested; refits nothing, re-tests nothing.
Every file is a ``booktabs`` body wrapped in ``threeparttable`` so that
``\\inputtable{<name>}`` (paper/main.tex) drops it straight into a floating
``table`` environment (``\\footnotesize``, ``\\tabcolsep`` 2.5pt, notes in
``\\scriptsize``). The one exception is ``tab-infeasible-full.tex``, a
``longtable`` fragment for the appendix that must be input OUTSIDE a float.
``tab-grid.tex`` is static and is never written here.

Scoring discipline enforced in code, not by convention
------------------------------------------------------
* Rows with ``scores_secondary`` (the conformal mechanisms) carry PLACEHOLDER
  CRPS / log score / PIT (uniform-in-interval samples; addendum 2 §3). They
  never enter a proper-score ranking, a PIT table or a PIT-Murphy column;
  they are compared on ``winkler_95`` / ``coverage_95`` only and their
  50 % / 80 % columns are printed as n/a (NaN by design in the runner).
  Their e0 / e65 / annuity quantiles are likewise computed by the runner from
  the uniform-in-interval filler samples, so ``tab-h5-actuarial`` prints
  those rows as n/a and never tabulates them.
* Arms with a different ``n_ages_scored`` are never averaged into the same
  ranking: CBD (ages 55-99) gets its own block with the support stated.
* Error rows enter no mean. They are counted per cell (``n_err``) and
  tabulated by QA class in ``tab-infeasible`` (compact, main text) and
  ``tab-infeasible-full`` (appendix longtable) using the regexes of
  ``scripts/final_qa.py``; a machine-failure row aborts table generation.
* A ranking (tab-h1) is a contrast, so it is computed on the common-cell
  intersection of addendum 3 §11 and reports kept / dropped units. The
  descriptive family x mechanism tables are uncensored and state so.
* Absent regimes (no parquet supplied) are printed as explicit ``pending``
  placeholders, never dropped silently. Snapshot inputs stamp every file
  with a NOT FINAL first line.
* The multi-output GP family runs in a SECOND pass (scripts/launch_sweeps.sh)
  and lands in ``results/<regime>_gp.parquet``; pass it as an additional
  ``--parquet`` and its rows merge by regime. Every family x mechanism table
  carries an explicit GP block; a regime without GP rows prints
  ``GP: pending (second-pass parquet)`` rather than omitting the family.
* ``tab-populations`` is generated FROM THE DATA FILES (HMD bulk Deaths_1x1
  and Exposures_1x1), never from a parquet, so it carries no snapshot stamp.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import final_qa                                                  # noqa: E402
from mortcal.runner import CONFORMAL_MECHANISMS, MECHANISMS, MODELS  # noqa: E402
from mortcal.splits import PLACEBO, PLACEBO_POPS, SHIFT, SHIFT_POPS  # noqa: E402

# ---------------------------------------------------------------------------
# registered constants and display names
# ---------------------------------------------------------------------------

#: Regimes the paper hooks expect (paper/sections/06-results.tex, 07-...).
EXPECTED_REGIMES: tuple[str, ...] = ("stable", "shift", "placebo")

FAMILY_ORDER: tuple[str, ...] = tuple(MODELS)
FAMILY_LABEL = {
    "LC": "Lee--Carter (SVD)", "PLC": "Poisson Lee--Carter", "CBD": "CBD (M5)",
    "RH": "APC / RH", "SVAR": "sparse VAR", "GP": "multi-output GP",
    "NLC": "neural-LC", "CNN": "CNN-LC", "LSTM": "LSTM-$\\kappa_t$",
    "NB": "NB head",
}
MECH_ORDER: tuple[str, ...] = tuple(MECHANISMS)
MECH_LABEL = {
    "native": "native", "pboot": "P-boot", "ensemble": "ensemble",
    "dropout": "MC-drop", "split_conf": "split", "enbpi": "EnbPI",
    "copula_conf": "copula",
}

#: PREREGISTRATION-ADDENDUM-1 §A strata (transcribed; the addendum is the
#: authority). Populations outside PLACEBO_POPS have stratum "none".
PLACEBO_STRATA: dict[str, str] = {
    **{p: "neutral" for p in ("CHE", "DNK", "FIN", "ISL", "NLD", "NOR", "SWE")},
    **{p: "belligerent" for p in ("FRATNP", "GBRTENW", "ITA")},
    "GBR_SCO": "civilian-only",
}
STRATA_ORDER = ("neutral", "belligerent", "civilian-only")

#: Families with a restricted age support (docs/GRID.md, runner MODEL_KWARGS).
#: Their rows are reported in their own block, never ranked with full-age arms.
RESTRICTED_AGE_FAMILIES: dict[str, str] = {"CBD": "ages 55--99"}

UNIT_KEYS = ["regime", "pop", "sex", "origin"]
CELL_KEYS = UNIT_KEYS + ["model", "mechanism"]

#: The second-pass family (scripts/launch_sweeps.sh pass 2; results/<regime>_gp.parquet).
GP_FAMILY = "GP"
GP_PENDING = "GP: \\emph{pending} (second-pass parquet)"

TABLE_NAMES = (
    "tab-populations", "tab-h1-rankings", "tab-h2-coverage", "tab-h3-joint",
    "tab-h4-age", "tab-h5-actuarial", "tab-murphy", "tab-pit", "tab-dm-mcs",
    "tab-twin-crises", "tab-infeasible", "tab-infeasible-full",
)

#: Default HMD bulk files (data/MANIFEST.sha256 pins the vintage). Absent on a
#: machine without the Dataset/ tree -> tab-populations prints pending cells.
DEFAULT_HMD_DEATHS = ROOT / "Dataset" / "deaths" / "Deaths_1x1" / "Deaths_1x1.txt"
DEFAULT_HMD_EXPOSURES = ROOT / "Dataset" / "exposures" / "Exposures_1x1" / "Exposures_1x1.txt"

# ---------------------------------------------------------------------------
# small formatting helpers
# ---------------------------------------------------------------------------

_LATEX_SPECIALS = {"_": "\\_", "%": "\\%", "&": "\\&", "#": "\\#", "$": "\\$",
                   "{": "\\{", "}": "\\}", "<": "$<$", ">": "$>$",
                   "~": "\\textasciitilde{}"}


def tex(s: object) -> str:
    """Escape a plain string for LaTeX (pop codes carry underscores)."""
    return "".join(_LATEX_SPECIALS.get(ch, ch) for ch in str(s))


def f3(x, nd: int = 3) -> str:
    """Fixed-point number or an em-dash where nothing was scored."""
    try:
        if x is None or pd.isna(x):
            return "--"
    except (TypeError, ValueError):
        return "--"
    return f"{float(x):.{nd}f}"


def fint(x) -> str:
    try:
        if x is None or pd.isna(x):
            return "--"
    except (TypeError, ValueError):
        return "--"
    return str(int(x))


PENDING = "\\emph{pending}"


def fam(m: str) -> str:
    return FAMILY_LABEL.get(m, tex(m))


def mech(u: str) -> str:
    return MECH_LABEL.get(u, tex(u))


def is_conformal(u: str) -> bool:
    return u in CONFORMAL_MECHANISMS


def _order_key(model: str, mechanism: str) -> tuple[int, int]:
    fi = FAMILY_ORDER.index(model) if model in FAMILY_ORDER else len(FAMILY_ORDER)
    mi = MECH_ORDER.index(mechanism) if mechanism in MECH_ORDER else len(MECH_ORDER)
    return fi, mi


def sort_cells(cells) -> list[tuple[str, str]]:
    return sorted(set(cells), key=lambda c: _order_key(*c))


# ---------------------------------------------------------------------------
# loading and classification
# ---------------------------------------------------------------------------

def regime_group(name: str) -> str:
    """'stable_1990' -> 'stable'; 'shift' -> 'shift'."""
    return str(name).split("_")[0]


def is_snapshot_path(p: str | Path) -> bool:
    name = Path(p).name
    return name.startswith("_") or "snapshot" in name.lower()


def load_rows(paths: list[str]) -> pd.DataFrame:
    """Concatenate every runner parquet (pass-1 files and the second-pass
    ``<regime>_gp.parquet`` alike); rows merge by their own ``regime`` column.
    A cell present in two inputs would be double-counted in every mean, so
    duplicates abort."""
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        df["_source"] = Path(p).name
        frames.append(df)
    if not frames:
        return prepare_rows(pd.DataFrame(columns=CELL_KEYS + ["error"]))
    df = pd.concat(frames, ignore_index=True)
    keys = [k for k in CELL_KEYS if k in df]
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        srcs = sorted(df.loc[dup, "_source"].unique())
        raise SystemExit(f"{int(dup.sum())} duplicate cell rows across inputs {srcs}; "
                         "merge by regime refused -- fix the parquets, never average twins")
    return prepare_rows(df)


def prepare_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise dtypes the tables rely on; add regime_group and error_class."""
    df = df.copy()
    if "error" not in df:
        df["error"] = None
    df["error"] = df["error"].astype(object).where(df["error"].notna(), None)
    if "scores_secondary" not in df:
        df["scores_secondary"] = False
    # error rows carry NaN here; the mechanism registry is the authority, so a
    # conformal error row can never be counted as a distributional cell
    df["scores_secondary"] = (df["scores_secondary"].fillna(False).astype(bool)
                              | df["mechanism"].isin(CONFORMAL_MECHANISMS))
    if "origin" not in df:
        df["origin"] = np.nan
    df["regime_group"] = df["regime"].map(regime_group)
    df["error_class"] = classify_errors(df["error"])
    return df


def classify_errors(err: pd.Series) -> pd.Series:
    """QA class per row using scripts/final_qa.py's regexes, same precedence:
    machine > structural > method > other; None for valid rows."""
    out = []
    for e in err:
        if e is None or (isinstance(e, float) and np.isnan(e)):
            out.append(None)
            continue
        s = str(e)
        if final_qa.MACHINE.search(s):
            out.append("machine")
        elif final_qa.STRUCTURAL.search(s):
            out.append("structural")
        elif final_qa.METHOD.search(s):
            out.append("method")
        else:
            out.append("other")
    return pd.Series(out, index=err.index, dtype=object)


def valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["error"].isna()]


def distributional(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["scores_secondary"]]


def load_analyses(paths: list[str]) -> dict[str, dict]:
    """regime_group -> analysis JSON (the JSON records its regimes)."""
    out = {}
    for p in paths:
        a = json.loads(Path(p).read_text(encoding="utf-8"))
        regs = a.get("regimes") or []
        key = regime_group(regs[0]) if regs else regime_group(Path(p).stem)
        a["_source"] = Path(p).name
        out[key] = a
    return out


def load_sensitivities(path: str | None) -> dict | None:
    if not path or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# common-cell restriction (addendum 3 §11)
# ---------------------------------------------------------------------------

def common_cells(df: pd.DataFrame, arms: list[tuple[str, str]]):
    """Units (regime, pop, sex, origin) on which EVERY arm has a valid row.

    Arms with no valid row anywhere are set aside (listed in the report)
    rather than allowed to empty the intersection. Returns
    (kept_units DataFrame, report dict).
    """
    ok = valid_rows(df)
    per_arm = {}
    for m, u in arms:
        sub = ok[(ok["model"] == m) & (ok["mechanism"] == u)]
        per_arm[(m, u)] = set(map(tuple, sub[UNIT_KEYS].itertuples(index=False)))
    usable = {a: s for a, s in per_arm.items() if s}
    absent = [a for a, s in per_arm.items() if not s]
    all_units = set(map(tuple, df[UNIT_KEYS].drop_duplicates().itertuples(index=False)))
    kept = set.intersection(*usable.values()) if usable else set()
    report = {
        "n_units_total": len(all_units),
        "n_kept": len(kept),
        "n_dropped": len(all_units) - len(kept),
        "arms_absent": absent,
        "arms_with_failures": [a for a, s in usable.items() if len(s) < len(all_units)],
    }
    kept_df = pd.DataFrame(sorted(kept), columns=UNIT_KEYS)
    return kept_df, report


def restrict(df: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    if units.empty:
        return df.iloc[0:0]
    return df.merge(units, on=UNIT_KEYS, how="inner")


# ---------------------------------------------------------------------------
# LaTeX assembly
# ---------------------------------------------------------------------------

class TableWriter:
    def __init__(self, out_dir: Path, sources: list[str], snapshot: bool):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.sources = sources
        self.snapshot = snapshot
        self.written: list[Path] = []

    def header(self, data_sources: list[str] | None = None) -> str:
        """First-line comment(s). ``data_sources`` marks a fragment generated
        from the HMD data files themselves (tab-populations): it is then not
        snapshot-derived and carries no NOT FINAL stamp."""
        lines = []
        today = _dt.date.today().isoformat()
        if data_sources is not None:
            lines.append("% GENERATED FROM THE DATA FILES (not snapshot-derived) by "
                         f"scripts/make_tables.py on {today} from "
                         f"{', '.join(data_sources)}; do not hand-edit.")
            return "\n".join(lines)
        if self.snapshot:
            regs = ", ".join(f"results/{r}.parquet + results/{r}_gp.parquet"
                             for r in EXPECTED_REGIMES)
            lines.append(f"% GENERATED SNAPSHOT - NOT FINAL - regenerate from {regs}")
        lines.append("% GENERATED by scripts/make_tables.py on "
                     f"{today} from "
                     f"{', '.join(self.sources) or '(no inputs)'}; do not hand-edit.")
        return "\n".join(lines)

    def write(self, name: str, colspec: str, header_rows: list[str],
              body_rows: list[str], notes: list[str], size: str = "\\scriptsize",
              tabcolsep: str = "2.5pt", data_sources: list[str] | None = None) -> Path:
        # \scriptsize + arraystretch 0.90: the family x mechanism x regime
        # tables (~55 body rows plus notes) overflowed a page by 80-120 pt at
        # \footnotesize (measured 2026-08-28); this brings them under a page
        # without migrating captions out of the sections into longtables.
        parts = [self.header(data_sources),
                 f"\\begin{{threeparttable}}{size}\\renewcommand{{\\arraystretch}}{{0.90}}"
                 f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}",
                 f"\\begin{{tabular}}{{@{{}}{colspec}@{{}}}}",
                 "\\toprule",
                 *header_rows,
                 "\\midrule",
                 *(body_rows or [f"\\multicolumn{{{_ncols(colspec)}}}{{l}}{{{PENDING}: no rows}} \\\\"]),
                 "\\bottomrule",
                 "\\end{tabular}",
                 "\\begin{tablenotes}\\scriptsize"]
        parts += [f"\\item {n}" for n in notes]
        parts += ["\\end{tablenotes}", "\\end{threeparttable}", ""]
        path = self.out_dir / f"{name}.tex"
        path.write_text("\n".join(parts), encoding="utf-8")
        self.written.append(path)
        return path

    def write_longtable(self, name: str, colspec: str, header_rows: list[str],
                        body_rows: list[str], notes: list[str], caption: str,
                        label: str, size: str = "\\footnotesize",
                        tabcolsep: str = "2.5pt") -> Path:
        """Appendix fragment: a ``longtable`` (package ``longtable``) that must
        be input outside any float. Notes follow as a ``\\scriptsize`` block."""
        ncols = _ncols(colspec)
        parts = [self.header(),
                 f"{{{size}\\setlength{{\\tabcolsep}}{{{tabcolsep}}}",
                 f"\\begin{{longtable}}{{@{{}}{colspec}@{{}}}}",
                 f"\\caption{{{caption}}}\\label{{{label}}} \\\\",
                 "\\toprule", *header_rows, "\\midrule", "\\endfirsthead",
                 f"\\multicolumn{{{ncols}}}{{l}}{{\\emph{{(continued)}}}} \\\\",
                 "\\toprule", *header_rows, "\\midrule", "\\endhead",
                 f"\\midrule \\multicolumn{{{ncols}}}{{r}}{{\\emph{{continued on next page}}}} \\\\",
                 "\\endfoot",
                 "\\bottomrule", "\\endlastfoot",
                 *(body_rows or [f"\\multicolumn{{{ncols}}}{{l}}{{{PENDING}: no rows}} \\\\"]),
                 "\\end{longtable}",
                 "\\begin{minipage}{\\linewidth}\\scriptsize"]
        parts += [f"\\noindent {n}\\par" for n in notes]
        parts += ["\\end{minipage}}", ""]
        path = self.out_dir / f"{name}.tex"
        path.write_text("\n".join(parts), encoding="utf-8")
        self.written.append(path)
        return path


def _ncols(colspec: str) -> int:
    return sum(ch in "lcr" for ch in re.sub(r"p\{[^}]*\}", "l", colspec))


def span_row(text: str, ncols: int, rule: bool = True) -> list[str]:
    row = f"\\multicolumn{{{ncols}}}{{l}}{{{text}}} \\\\"
    return [row, "\\midrule"] if rule else [row]


def pending_row(regime: str, ncols: int, why: str = "no parquet supplied") -> str:
    return (f"\\multicolumn{{{ncols}}}{{l}}{{{tex(regime)} regime: {PENDING} "
            f"({why})}} \\\\")


def gp_pending_cell(width: int) -> str:
    """One regime block's worth of columns saying the GP pass is outstanding."""
    return f"\\multicolumn{{{width}}}{{l}}{{{GP_PENDING}}}"


def gp_present(df: pd.DataFrame) -> bool:
    """True when the (regime) frame carries any GP row, valid or error."""
    return bool(len(df)) and bool((df["model"] == GP_FAMILY).any())


def gp_note(df: pd.DataFrame) -> str:
    """Tablenote listing the regimes whose GP block is pending."""
    missing = [r for r in EXPECTED_REGIMES
               if not regime_frame(df, r).empty and not gp_present(regime_frame(df, r))]
    if not missing:
        return ("Multi-output GP rows come from the second-pass parquet "
                "(\\texttt{results/<regime>\\_gp.parquet}) merged by regime.")
    return ("Multi-output GP runs in a second pass (\\texttt{results/<regime>\\_gp.parquet}); "
            f"no GP rows supplied for {', '.join(missing)}: {GP_PENDING}.")


# ---------------------------------------------------------------------------
# regime bookkeeping for the notes
# ---------------------------------------------------------------------------

def regime_frame(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    return df[df["regime_group"] == regime]


def present_regimes(df: pd.DataFrame) -> list[str]:
    present = set(df["regime_group"].unique()) if len(df) else set()
    return [r for r in EXPECTED_REGIMES if r in present] + sorted(
        r for r in present if r not in EXPECTED_REGIMES)


def regime_note(df: pd.DataFrame, regime: str, extra: str = "") -> str:
    sub = regime_frame(df, regime)
    if sub.empty:
        return f"{regime}: {PENDING} -- no parquet supplied for this regime."
    n_pop = sub["pop"].nunique()
    n_units = len(sub[UNIT_KEYS].drop_duplicates())
    n_origin = sub["origin"].nunique()
    n_err = int(sub["error"].notna().sum())
    txt = (f"{regime}: {n_pop} populations, {n_units} (population, sex, origin) "
           f"units over {n_origin} origin(s), {len(sub)} cell rows of which "
           f"{n_err} error rows are excluded from every mean.")
    return txt + (" " + extra if extra else "")


DESCRIPTIVE_NOTE = ("Descriptive table: no common-cell restriction is applied "
                    "(addendum 3 \\S11 applies to contrasts); $n$ is the number "
                    "of valid rows entering each mean and $n_{\\mathrm{err}}$ the "
                    "error rows of that cell excluded from it.")
CONFORMAL_NOTE = ("Conformal mechanisms (split, EnbPI, copula) construct one "
                  "interval at 95\\%; their 50\\%/80\\% columns are not "
                  "applicable (n/a) by design and their CRPS / log score / PIT "
                  "are placeholders never tabulated as proper scores "
                  "(addendum 2 \\S3).")
CBD_NOTE = "CBD (M5) is scored on ages 55--99 only; its rows are not comparable with full-age (0--99) families."


def cell_stats(ok: pd.DataFrame, err: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Mean of `cols` and n per (model, mechanism) over valid rows; n_err from
    error rows. Means use only finite values (NaN-aware)."""
    if ok.empty and err.empty:
        return pd.DataFrame(columns=["model", "mechanism", "n", "n_err", *cols])
    g = ok.groupby(["model", "mechanism"], sort=False)
    stats = g[cols].mean(numeric_only=True) if not ok.empty else pd.DataFrame(columns=cols)
    stats["n"] = g.size() if not ok.empty else 0
    ne = err.groupby(["model", "mechanism"]).size().rename("n_err") if not err.empty else None
    stats = stats.reset_index()
    if ne is not None:
        stats = stats.merge(ne.reset_index(), on=["model", "mechanism"], how="outer")
        missing = stats["n"].isna()
        stats.loc[missing, "n"] = 0
    else:
        stats["n_err"] = 0
    stats["n_err"] = stats["n_err"].fillna(0).astype(int)
    stats["n"] = stats["n"].astype(int)
    stats["_k"] = [_order_key(m, u) for m, u in zip(stats["model"], stats["mechanism"])]
    return stats.sort_values("_k").drop(columns="_k").reset_index(drop=True)


def _lookup(stats: pd.DataFrame, m: str, u: str) -> pd.Series | None:
    hit = stats[(stats["model"] == m) & (stats["mechanism"] == u)]
    return hit.iloc[0] if len(hit) else None


# ---------------------------------------------------------------------------
# generic "family x mechanism, one column block per regime" layout
# ---------------------------------------------------------------------------

def block_table(df: pd.DataFrame, cols: list[str], col_heads: list[str],
                fmt, regimes: list[str] | None = None, rows_filter=None,
                per_block_stats=None):
    """Return (colspec, header_rows, body_rows).

    `fmt(row_stats, model, mechanism, col)` -> cell string. Absent regimes
    get a one-column pending block. `rows_filter(df)` narrows the rows used.
    """
    regimes = regimes or list(EXPECTED_REGIMES)
    frames = {r: regime_frame(df, r) for r in regimes}
    if rows_filter is not None:
        frames = {r: rows_filter(f) for r, f in frames.items()}
    stats = {}
    for r, f in frames.items():
        if f.empty:
            stats[r] = None
            continue
        stats[r] = (per_block_stats or cell_stats)(valid_rows(f), f[f["error"].notna()], cols)
    cells = sort_cells((m, u) for s in stats.values() if s is not None
                       for m, u in zip(s["model"], s["mechanism"]))
    # explicit GP block: the family is never silently absent (second pass)
    gp_here = {r: gp_present(f) for r, f in frames.items()}
    if not any(m == GP_FAMILY for m, _ in cells):
        cells = sort_cells([*cells, (GP_FAMILY, "")])
    colspec = "ll"
    head1 = ["Family", "Mech."]
    head2 = ["", ""]
    for r in regimes:
        if stats[r] is None:
            colspec += "c"
            head1.append(f"\\multicolumn{{1}}{{c}}{{{r}}}")
            head2.append(PENDING)
        else:
            colspec += "r" * (len(cols) + 2)
            head1.append(f"\\multicolumn{{{len(cols) + 2}}}{{c}}{{{r}}}")
            head2 += [*col_heads, "$n$", "$n_{\\mathrm{err}}$"]
    header_rows = [" & ".join(head1) + " \\\\", " & ".join(head2) + " \\\\"]
    body = []
    last_fam = None
    for m, u in cells:
        if last_fam is not None and m != last_fam:
            body.append("\\addlinespace[2pt]")
        line = [fam(m) if m != last_fam else "", mech(u) if u else "all arms"]
        last_fam = m
        for r in regimes:
            s = stats[r]
            if s is None:
                line.append("--")
                continue
            if m == GP_FAMILY and not gp_here[r]:
                line.append(gp_pending_cell(len(cols) + 2))
                continue
            row = _lookup(s, m, u) if u else None
            if row is None:
                line += ["--"] * len(cols) + ["0", "0"]
                continue
            line += [fmt(row, m, u, c) for c in cols]
            line += [fint(row["n"]), fint(row["n_err"])]
        body.append(" & ".join(line) + " \\\\")
    return colspec, header_rows, body


def fmt_levels(row, m, u, col):
    """n/a for the 50/80 columns of a conformal row; number otherwise."""
    if is_conformal(u) and re.search(r"(50|80)", col) and "95" not in col:
        return "n/a"
    return f3(row[col])


# ---------------------------------------------------------------------------
# the eleven tables
# ---------------------------------------------------------------------------

AGE_MAX = 99            # the study models ages 0..99 (mortcal.data.hmd.build_panel)
ZERO_DEATH = 0.5        # addendum 3 §10: D < 0.5 is a zero-death cell


def _training_years(years_present: list[int], origin: int) -> list[int]:
    """Runner rule (addendum 3 §2): maximal contiguous block of years present
    in the panel ending at the origin (a year is present when the pivot has
    the column, i.e. any age carries D and E)."""
    train = sorted(y for y in years_present if y <= origin)
    gaps = np.flatnonzero(np.diff(np.asarray(train)) > 1)
    if len(gaps):
        train = train[gaps[-1] + 1:]
    return train


def population_facts(deaths_path: str | Path | None,
                     exposures_path: str | Path | None,
                     pops: tuple[str, ...] = SHIFT_POPS,
                     regime=SHIFT) -> dict[str, dict]:
    """Per population, straight from the HMD bulk files (no parquet):

    first_year / last_year  -- first and last year of single-age data
    n_train                 -- training years to the origin (min over sexes)
    zero_E_{f,m}            -- zero-EXPOSURE cells (E == 0 or missing) at ages
                               0..99 in the training panel, by sex
    zero_D_{f,m}            -- zero-death cells (D < 0.5) at ages 0..99 in the
                               test window, by sex (addendum 3 §10)

    Empty when either file is missing (every cell is then printed pending).
    """
    if not deaths_path or not exposures_path:
        return {}
    if not Path(deaths_path).exists() or not Path(exposures_path).exists():
        return {}
    from mortcal.data.hmd import read_bulk_1x1
    d = read_bulk_1x1(deaths_path, list(pops))
    e = read_bulk_1x1(exposures_path, list(pops))
    d = d[d["age"] <= AGE_MAX]
    e = e[e["age"] <= AGE_MAX]
    test = list(regime.test_years)
    out = {}
    for p in sorted(set(d["pop"]) | set(e["pop"])):
        dp, ep = d[d["pop"] == p], e[e["pop"] == p]
        years = sorted(set(dp["year"].astype(int)) | set(ep["year"].astype(int)))
        rec = {"first_year": years[0] if years else None,
               "last_year": years[-1] if years else None}
        n_train = []
        for sex, tag in (("female", "f"), ("male", "m")):
            m = dp[["year", "age", sex]].rename(columns={sex: "D"}).merge(
                ep[["year", "age", sex]].rename(columns={sex: "E"}),
                on=["year", "age"], how="outer")
            present = m[m["D"].notna() & m["E"].notna()]
            train = _training_years(sorted(set(present["year"].astype(int))),
                                    regime.train_max_year)
            n_train.append(len(train))
            # training panel = every (year, age) of the block, so a row that is
            # absent from the file altogether also counts as missing exposure
            tr = m[m["year"].isin(train)] if train else m.iloc[0:0]
            n_cells_expected = len(train) * (AGE_MAX + 1)
            zero_e = int(((tr["E"].isna()) | (tr["E"] == 0)).sum()) + (n_cells_expected - len(tr))
            rec[f"zero_E_{tag}"] = zero_e if train else None
            te = m[m["year"].isin(test)]
            rec[f"zero_D_{tag}"] = int((te["D"] < ZERO_DEATH).sum()) if len(te) else None
        rec["n_train"] = min(n_train) if n_train else None
        out[p] = rec
    return out


def tab_populations(w: TableWriter, hmd_deaths: str | Path | None,
                    hmd_exposures: str | Path | None):
    """Generated from the data files (paper/sections/03-data.tex caption):
    no parquet, no snapshot stamp."""
    facts = population_facts(hmd_deaths, hmd_exposures)
    pops = list(SHIFT_POPS)
    body = []

    def cell(rec, key):
        return PENDING if rec is None or rec.get(key) is None else fint(rec[key])

    for p in pops:
        rec = facts.get(p)
        body.append(" & ".join([
            tex(p), cell(rec, "first_year"), cell(rec, "last_year"), cell(rec, "n_train"),
            cell(rec, "zero_E_f"), cell(rec, "zero_E_m"), cell(rec, "zero_D_f"), cell(rec, "zero_D_m"),
            "yes" if p in PLACEBO_POPS else "no", PLACEBO_STRATA.get(p, "none")]) + " \\\\")
    placebo_only = [p for p in PLACEBO_POPS if p not in pops]
    if placebo_only:
        body.append("\\midrule")
        body += span_row("Placebo-only populations (train $\\le$1913, test 1914--1922)", 10, rule=False)
        for p in placebo_only:
            body.append(" & ".join([tex(p), "--", "--", "--", "--", "--", "--", "--", "yes",
                                    PLACEBO_STRATA.get(p, "none")]) + " \\\\")
    src = ("the HMD bulk \\texttt{Deaths\\_1x1} and \\texttt{Exposures\\_1x1} files "
           "pinned in \\texttt{data/MANIFEST.sha256}" if facts
           else f"{PENDING} until the HMD \\texttt{{Deaths\\_1x1}} and "
                "\\texttt{Exposures\\_1x1} bulk files are supplied")
    notes = [
        f"Generated from the data files -- {src} -- not from any model output; "
        f"shift populations per PREREGISTRATION.md ({len(pops)}).",
        "First / last year: first and last year of single-age data for the population. "
        "Train yrs: maximal contiguous block of years with data ending at the 2019 "
        "origin, minimum over sexes (addendum 3 \\S2; BEL trains 1919--2019).",
        f"Zero-$E$: training-panel cells at ages 0--{AGE_MAX} with $E=0$ or missing, by "
        f"sex (addendum 3 \\S1). Zero-$D$: cells with $D<{ZERO_DEATH}$ at ages 0--{AGE_MAX} "
        f"in the {SHIFT.test_years[0]}--{SHIFT.test_years[-1]} test window before any "
        "model age mask, by sex (addendum 3 \\S10).",
        "Placebo eligibility and stratum per PREREGISTRATION-ADDENDUM-1 \\S A: "
        "neutral / register-based (CHE, DNK, FIN, ISL, NLD, NOR, SWE), belligerent "
        "total series (FRATNP, GBRTENW, ITA), civilian-only (GBR\\_SCO); BEL excluded "
        "(missing 1914--1918).",
    ]
    data_sources = ([Path(hmd_deaths).name, Path(hmd_exposures).name] if facts
                    else ["(HMD bulk files not supplied)"])
    w.write("tab-populations", "lrrrrrrrll",
            ["Pop. & First & Last & Train yrs & \\multicolumn{2}{c}{Zero-$E$ train} & "
             "\\multicolumn{2}{c}{Zero-$D$ test} & Placebo & Stratum \\\\",
             " & year & year & $\\le$2019 & F & M & F & M & & \\\\"],
            body, notes, data_sources=data_sources)


H1_COLS = ["rmse_logmx", "crps_logmx", "poisson_log_score"]


def _rank_block(ok: pd.DataFrame, arms: list[tuple[str, str]]):
    """Mean of the three scores per arm on the common cells + ranks (1 = best;
    all three are negatively oriented) and Spearman correlations."""
    kept, rep = common_cells(ok, arms)
    sub = restrict(ok, kept)
    sub = sub[[ (m, u) in set(arms) for m, u in zip(sub["model"], sub["mechanism"])]]
    if sub.empty:
        return None, rep
    st = sub.groupby(["model", "mechanism"])[H1_COLS].mean().reset_index()
    for c in H1_COLS:
        st[f"rank_{c}"] = st[c].rank(method="min").astype(int)
    st["_k"] = [_order_key(m, u) for m, u in zip(st["model"], st["mechanism"])]
    st = st.sort_values("_k").drop(columns="_k").reset_index(drop=True)
    rho = {}
    for c in ("crps_logmx", "poisson_log_score"):
        a, b = st["rank_rmse_logmx"], st[f"rank_{c}"]
        if len(st) >= 3 and a.nunique() > 1 and b.nunique() > 1:
            rho[c] = float(a.corr(b, method="spearman"))
        else:
            rho[c] = np.nan   # undefined for < 3 arms or a constant rank vector
    return (st, rho), rep


def tab_h1(df: pd.DataFrame, w: TableWriter):
    ncols = 8
    body, notes = [], []
    for r in EXPECTED_REGIMES:
        sub = regime_frame(df, r)
        if sub.empty:
            body.append(pending_row(r, ncols))
            notes.append(regime_note(df, r))
            continue
        ok = distributional(valid_rows(sub))
        all_arms = sort_cells((m, u) for m, u in zip(ok["model"], ok["mechanism"]))
        main_arms = [a for a in all_arms if a[0] not in RESTRICTED_AGE_FAMILIES]
        body += span_row(f"\\textbf{{{r} regime}} -- full-age families (0--99), distributional mechanisms", ncols)
        if not gp_present(sub):
            body.append(f"\\multicolumn{{{ncols}}}{{l}}{{{fam(GP_FAMILY)} -- {GP_PENDING}; "
                        f"ranks below exclude it}} \\\\")
        elif not any(m == GP_FAMILY for m, _ in all_arms):
            body.append(f"\\multicolumn{{{ncols}}}{{l}}{{{fam(GP_FAMILY)} -- no valid "
                        "distributional cell (see tab-infeasible)}} \\\\")
        res, rep = _rank_block(sub[~sub["scores_secondary"]], main_arms)
        if res is None:
            body.append(pending_row(r, ncols, "no common cell across the distributional arms"))
        else:
            st, rho = res
            for _, x in st.iterrows():
                body.append(" & ".join([
                    fam(x["model"]), mech(x["mechanism"]),
                    f3(x["rmse_logmx"]), fint(x["rank_rmse_logmx"]),
                    f3(x["crps_logmx"]), fint(x["rank_crps_logmx"]),
                    f3(x["poisson_log_score"], 2), fint(x["rank_poisson_log_score"])]) + " \\\\")
            notes.append(
                f"{r}, main block: Spearman $\\rho$(rank RMSE, rank CRPS) = {f3(rho['crps_logmx'], 2)}, "
                f"$\\rho$(rank RMSE, rank log score) = {f3(rho['poisson_log_score'], 2)}; "
                f"common-cell restriction (addendum 3 \\S11): {rep['n_kept']} of "
                f"{rep['n_units_total']} units kept, {rep['n_dropped']} dropped"
                + (f"; arms with failures: {', '.join(f'{m}/{u}' for m, u in rep['arms_with_failures'])}"
                   if rep["arms_with_failures"] else "")
                + (f"; arms with no valid cell: {', '.join(f'{m}/{u}' for m, u in rep['arms_absent'])}"
                   if rep["arms_absent"] else "") + ".")
        for famname, support in RESTRICTED_AGE_FAMILIES.items():
            arms = [a for a in all_arms if a[0] == famname]
            if not arms:
                continue
            body += span_row(f"\\emph{{{fam(famname)}, {support}, separate ranking (not comparable with the block above)}}", ncols)
            res, rep = _rank_block(sub[~sub["scores_secondary"]], arms)
            if res is None:
                body.append(pending_row(r, ncols, f"no common cell for {famname}"))
                continue
            st, rho = res
            for _, x in st.iterrows():
                body.append(" & ".join([
                    fam(x["model"]), mech(x["mechanism"]),
                    f3(x["rmse_logmx"]), fint(x["rank_rmse_logmx"]),
                    f3(x["crps_logmx"]), fint(x["rank_crps_logmx"]),
                    f3(x["poisson_log_score"], 2), fint(x["rank_poisson_log_score"])]) + " \\\\")
            notes.append(f"{r}, {famname} block: {rep['n_kept']} of {rep['n_units_total']} units kept "
                         f"({rep['n_dropped']} dropped); rank correlations are not reported "
                         f"for a block of {len(st)} arms.")
        notes.append(regime_note(df, r))
    notes += [
        "Ranks: 1 = best; RMSE on log rates, CRPS on log rates and the Poisson log "
        "score (negative log predictive density of rounded deaths) are all "
        "negatively oriented. Means are per row (population, sex, origin), "
        "unweighted by exposure (addendum 3 \\S8).",
        "Conformal rows are excluded: their CRPS / log score are placeholders "
        "(addendum 2 \\S3). " + CBD_NOTE,
        gp_note(df),
    ]
    w.write("tab-h1-rankings", "llrrrrrr",
            ["Family & Mech. & RMSE & rk & CRPS & rk & log score & rk \\\\"],
            body, notes)


def tab_h2(df: pd.DataFrame, w: TableWriter):
    cols = ["coverage_50", "coverage_80", "coverage_95", "winkler_95"]
    colspec, head, body = block_table(df, cols, ["cov$_{50}$", "cov$_{80}$", "cov$_{95}$", "$\\IS_{95}$"], fmt_levels)
    notes = [regime_note(df, r) for r in EXPECTED_REGIMES]
    notes += [DESCRIPTIVE_NOTE, CONFORMAL_NOTE, CBD_NOTE, gp_note(df),
              "$\\IS_{95}$: mean Winkler / interval score of the 95\\% interval on log rates (negatively oriented)."]
    w.write("tab-h2-coverage", colspec, head, body, notes)


def tab_h3(df: pd.DataFrame, w: TableWriter):
    cols = ["coverage_95", "joint_path_coverage_95", "gap"]

    def stats_with_gap(ok, err, cols_):
        st = cell_stats(ok, err, ["coverage_95", "joint_path_coverage_95"])
        st["gap"] = st["joint_path_coverage_95"] - st["coverage_95"]
        return st

    colspec, head, body = block_table(df, cols, ["marginal", "joint path", "gap"],
                                      lambda row, m, u, c: f3(row[c]), per_block_stats=stats_with_gap)
    notes = [regime_note(df, r) for r in EXPECTED_REGIMES]
    notes += [DESCRIPTIVE_NOTE,
              "Marginal = mean per-cell 95\\% coverage over horizons and scored ages; "
              "joint path = share of scored ages whose whole $h=1\\ldots H$ trajectory "
              "lies inside the 95\\% band; gap = joint $-$ marginal. Conformal rows use "
              "the wrapper's own interval bounds (addendum 3 \\S6); the copula arm is the "
              "only mechanism that constructs a joint band.", CBD_NOTE, gp_note(df)]
    w.write("tab-h3-joint", colspec, head, body, notes)


def tab_h4(df: pd.DataFrame, w: TableWriter):
    cols = ["coverage_95_band0_24", "coverage_95_band25_64", "coverage_95_band65_99"]
    colspec, head, body = block_table(df, cols, ["0--24", "25--64", "65--99"],
                                      lambda row, m, u, c: f3(row[c]))
    notes = [regime_note(df, r) for r in EXPECTED_REGIMES]
    notes += [DESCRIPTIVE_NOTE,
              "Empirical coverage of the nominal 95\\% interval by age band (runner "
              "AGE\\_BANDS; the last band is open above). CBD has no scored ages "
              "below 55, so its 0--24 band is empty and its 25--64 band covers "
              "55--64 only.",
              "Registered direction (worse at 65--99 than 25--64) is contradicted "
              "by Dowd et al.; a reversal is reported as informative (addendum 3 \\S8).",
              gp_note(df)]
    w.write("tab-h4-age", colspec, head, body, notes)


def _h5_stats(ok: pd.DataFrame, err: pd.DataFrame, cols_: list[str]) -> pd.DataFrame:
    """Per cell: empirical 95% coverage (share of rows with q025 <= obs <= q975,
    over rows where all three are finite) and mean error of e0, e65, ann65."""
    recs = []
    keys = set(map(tuple, ok[["model", "mechanism"]].drop_duplicates().itertuples(index=False)))
    keys |= set(map(tuple, err[["model", "mechanism"]].drop_duplicates().itertuples(index=False)))
    for m, u in sort_cells(keys):
        sub = ok[(ok["model"] == m) & (ok["mechanism"] == u)]
        rec = {"model": m, "mechanism": u, "n": len(sub),
               "n_err": int(((err["model"] == m) & (err["mechanism"] == u)).sum())}
        # conformal rows: the runner's e0/e65/ann65 quantiles come from the
        # uniform-in-interval filler samples, not a predictive distribution --
        # never computed here, printed n/a (addendum 2 §3)
        conformal = is_conformal(u) or (len(sub) > 0 and bool(sub["scores_secondary"].any()))
        for q in ("e0", "e65", "ann65"):
            if conformal:
                rec[f"{q}_cov"] = np.nan
                rec[f"{q}_err"] = np.nan
                rec[f"{q}_n"] = 0
                continue
            lo, hi, ob, er = (sub.get(f"{q}_{s}", pd.Series(dtype=float)) for s in ("q025", "q975", "obs", "error"))
            fin = lo.notna() & hi.notna() & ob.notna()
            rec[f"{q}_cov"] = float(((lo[fin] <= ob[fin]) & (ob[fin] <= hi[fin])).mean()) if fin.any() else np.nan
            rec[f"{q}_err"] = float(er.mean()) if er.notna().any() else np.nan
            rec[f"{q}_n"] = int(fin.sum())
        recs.append(rec)
    return pd.DataFrame(recs)


H5_CONFORMAL_NOTE = ("Conformal rows (split, EnbPI, copula) are n/a: derived-quantity "
                     "intervals require predictive samples; conformal mechanisms yield "
                     "interval bounds on $\\log m_x$ only (addendum 2 \\S3).")


def _fmt_h5(row, m, u, c):
    if is_conformal(u):
        return "n/a"
    return f3(row[c], 2 if c.endswith("_err") else 3)


def tab_h5(df: pd.DataFrame, w: TableWriter):
    cols = ["e0_cov", "e0_err", "e65_cov", "e65_err", "ann65_cov", "ann65_err"]
    heads = ["$e_0$ cov", "err", "$e_{65}$ cov", "err", "$\\annuity$ cov", "err"]
    colspec, head, body = block_table(df, cols, heads, _fmt_h5, per_block_stats=_h5_stats)
    notes = [regime_note(df, r) for r in EXPECTED_REGIMES]
    notes += [DESCRIPTIVE_NOTE,
              "cov = share of rows whose realised $e_0$, $e_{65}$ or $\\annuity$ (2\\%) "
              "lies inside the model's [2.5\\%, 97.5\\%] sample quantiles; err = mean "
              "(point $-$ observed), years for $e_x$, annuity units for $\\annuity$. "
              "Derived quantities are integrated from the LATENT predictive paths on "
              "the maximal contiguous scored age block from age 0 (addendum 3 \\S3).",
              H5_CONFORMAL_NOTE,
              "CBD's block starts at age 55: $e_0$ is undefined (--) and $e_{65}$ / "
              "$\\annuity$ come from a 55--99 table. Rows whose bounds are missing "
              "are excluded from that quantity's share only.", gp_note(df)]
    # longtable: three regimes x every family/mechanism plus the notes overflow
    # a single float page (measured +25 pt at \scriptsize, 2026-08-28), so the
    # fragment spans pages and carries its own caption/label.
    w.write_longtable(
        "tab-h5-actuarial", colspec, head, body, notes,
        caption=(r"Empirical coverage of the nominal 95\% interval---the share of "
                 r"population--sex cells in which the realised value lies between the "
                 r"2.5\% and 97.5\% sample quantiles---and the mean error (median of the "
                 r"samples minus the realised value) for $e_0$, $e_{65}$ and $\annuity$ "
                 r"at 2\%, by family, mechanism and regime, on the period table of the "
                 r"first test year. Conformal rows are reported as not applicable: the "
                 r"functionals require predictive sample paths, and conformal mechanisms "
                 r"yield interval bounds on $\log m_{x,t}$ only (Addendum~2~\S3). CBD "
                 r"(ages 55--99) has no $e_0$; its $e_{65}$ and $\annuity$ are exact on "
                 r"the truncated table. Regime, population count and the age range of "
                 r"the table are stated in the note."),
        label="tab:h5-actuarial")


def tab_murphy(df: pd.DataFrame, w: TableWriter):
    cols = ["murphy_reliability", "murphy_resolution", "murphy_uncertainty", "murphy_pit_reliability"]

    def fmt(row, m, u, c):
        if c == "murphy_pit_reliability" and is_conformal(u):
            return "n/a"
        return f3(row[c], 4)

    colspec, head, body = block_table(df, cols, ["REL", "RES", "UNC", "PIT-REL"], fmt)
    notes = [regime_note(df, r) for r in EXPECTED_REGIMES]
    notes += [DESCRIPTIVE_NOTE,
              "Murphy (1973) partition of the Brier score of the 95\\% hit indicator "
              "into reliability (REL, miscalibration), resolution (RES) and "
              "uncertainty (UNC): Brier $=$ REL $-$ RES $+$ UNC. With a single "
              "constant nominal level (0.95) as the forecast probability there is "
              "one bin, so RES is 0 by construction and REL $=(0.95-\\bar{y})^2$; "
              "the decomposition becomes informative only across levels. PIT-REL "
              "is the reliability term of the PIT-histogram decomposition "
              "(distributional rows only; n/a for conformal rows).", CBD_NOTE, gp_note(df)]
    w.write("tab-murphy", colspec, head, body, notes)


def _pit_stats(ok: pd.DataFrame, err: pd.DataFrame, cols_: list[str]) -> pd.DataFrame:
    st = cell_stats(ok, err, ["pit_ks_stat"])
    share = (ok.assign(sig=ok["pit_ks_pvalue"] < 0.05)
               .groupby(["model", "mechanism"])["sig"].mean().rename("share_p05").reset_index())
    st = st.merge(share, on=["model", "mechanism"], how="left")
    return st


def tab_pit(df: pd.DataFrame, w: TableWriter):
    cols = ["pit_ks_stat", "share_p05"]
    colspec, head, body = block_table(df, cols, ["KS", "share $p<0.05$"],
                                      lambda row, m, u, c: f3(row[c]),
                                      rows_filter=distributional, per_block_stats=_pit_stats)
    notes = [regime_note(df, r) for r in EXPECTED_REGIMES]
    notes += [DESCRIPTIVE_NOTE,
              "Distributional mechanisms only: conformal PIT values are placeholders "
              "(addendum 2 \\S3) and are omitted, not shown as n/a.",
              "KS = mean Kolmogorov--Smirnov distance of the randomised PIT from "
              "U(0,1) within a row; share $p<0.05$ = fraction of rows whose nominal "
              "KS $p$-value falls below 0.05. The $p$-value assumes independent "
              "draws; PIT values across ages and horizons within a population are "
              "dependent, so the share is descriptive (addendum 2 \\S4), never a test.",
              CBD_NOTE, gp_note(df)]
    w.write("tab-pit", colspec, head, body, notes)


def _fmt_pvals(pv: dict) -> str:
    return "; ".join(f"{tex(k)} {f3(v, 2)}" for k, v in pv.items())


def _skip_reason(v: dict) -> str:
    """First sentence of an analysis-stage skip reason (the age-support guard
    or an empty intersection), escaped for LaTeX."""
    return tex(str(v["skipped"]).split(". ")[0].rstrip("."))


def tab_dm_mcs(analyses: dict[str, dict], w: TableWriter):
    ncols = 6
    body, notes = [], []
    for r in EXPECTED_REGIMES:
        a = analyses.get(r)
        if a is None:
            body.append(pending_row(r, ncols, "no analysis JSON supplied"))
            notes.append(f"{r}: {PENDING} -- no analysis JSON.")
            continue
        body += span_row(f"\\textbf{{{r} regime}} -- model confidence sets ($1-\\alpha$ = {f3(1 - a.get('alpha', 0.10), 2)})", ncols)
        mcs_keys = [k for k in a if k.startswith("mcs_")]
        if not mcs_keys:
            body.append(pending_row(r, ncols, "no MCS contrasts in the JSON"))
        for k in mcs_keys:
            v = a[k]
            name = tex(k[4:])
            if "skipped" in v:
                body.append(f"{name} & -- & \\multicolumn{{4}}{{p{{9.6cm}}}}{{skipped: {_skip_reason(v)}}} \\\\")
                continue
            rep = v.get("intersection", {})
            body.append(" & ".join([
                name, tex(rep.get("loss", a.get("loss", "?"))),
                tex(", ".join(v.get("in_set", []))) or "--",
                _fmt_pvals(v.get("p_values", {})),
                fint(rep.get("n_cells_kept")), fint(rep.get("n_cells_dropped"))]) + " \\\\")
        body += span_row(f"\\textbf{{{r} regime}} -- Diebold--Mariano, native vs split conformal (wild cluster bootstrap)", ncols)
        dm = a.get("dm_native_vs_split", {})
        if not dm:
            body.append(pending_row(r, ncols, "no DM contrasts in the JSON"))
        for famname, v in dm.items():
            if "skipped" in v:
                body.append(f"{fam(famname)} & -- & \\multicolumn{{4}}{{p{{9.6cm}}}}{{skipped: {_skip_reason(v)}}} \\\\")
                continue
            rep = v.get("intersection", {})
            body.append(" & ".join([
                fam(famname), tex(rep.get("loss", "?")),
                f"$\\Delta$ = {f3(v.get('mean_diff'), 3)} (se {f3(v.get('se'), 3)})",
                f"$p$ = {f3(v.get('p_value'), 3)}, $G$ = {fint(v.get('n_clusters'))}",
                fint(rep.get("n_cells_kept")), fint(rep.get("n_cells_dropped"))]) + " \\\\")
        notes.append(f"{r}: analysis {tex(a.get('_source', ''))}, {a.get('n_rows', '?')} rows, "
                     f"{a.get('n_error_rows', '?')} error rows; default loss {tex(a.get('loss', '?'))}.")
    notes += [
        "Loss column: \\texttt{crps} / \\texttt{logscore} are per-horizon proper "
        "scores and are used only between distributional arms; every contrast "
        "that includes a conformal arm uses \\texttt{winkler95}, the per-horizon "
        "interval score at the construction level (addendum 2 \\S3). "
        "MCS $p$-values are the sequential elimination $p$-values (Hansen et al.); "
        "DM $\\Delta$ = mean loss(native) $-$ loss(split), negative favours native; "
        "$G$ = number of population clusters.",
        "Kept / dropped: (population, sex, origin) units on the common-cell "
        "intersection of the compared arms (addendum 3 \\S11). A contrast whose "
        "arms do not share an age support (CBD, 45 ages) is skipped, not forced.",
    ]
    # longtable: the MCS and DM blocks with wrapped p{} columns overflow a
    # float page by ~110 pt (measured 2026-08-28); the fragment spans pages
    # and carries its own caption/label.
    w.write_longtable(
        "tab-dm-mcs", "llp{3.2cm}p{5.2cm}rr",
        ["Contrast & Loss & In set / statistic & $p$-values & kept & dropped \\\\"],
        body, notes,
        caption=(r"Registered forecast-comparison tests from the analysis stage. "
                 r"Upper block: each model confidence set at 90\%---contrast name, loss "
                 r"used, arms retained, elimination $p$-values, and cells kept and "
                 r"dropped by the common-cell restriction. Lower block: pairwise "
                 r"Diebold--Mariano, native versus split conformal within each "
                 r"family---loss used, mean loss differential, wild-cluster-bootstrap "
                 r"$p$-value and number of clusters. Contrasts among distributional arms "
                 r"use CRPS on log rates; every contrast that includes a conformal arm "
                 r"uses the per-horizon 95\% interval score. A contrast that cannot be "
                 r"formed is listed with its recorded reason."),
        label="tab:dm-mcs")


#: scripts/sensitivities.py slice names -> the addendum-1 stratum labels used here
SENS_STRATA_KEYS = {"neutral": "neutral", "belligerent_total": "belligerent",
                    "civilian_only": "civilian-only"}


def _strata_from_sensitivities(sens: dict | None) -> dict[str, pd.DataFrame] | None:
    """Read the scripts/sensitivities.py contract (its module docstring):
    ``sens["strata"]["placebo"]`` is the literal ``"pending"`` while the
    placebo parquet is absent, else ``{slice: {"_meta": ..., "FAM/MECH":
    {"coverage_95", "joint_path_coverage_95", "n_cells", ...}}}``."""
    block = (sens or {}).get("strata", {}).get("placebo") if sens else None
    if not isinstance(block, dict):
        return None
    out = {}
    for key, label in SENS_STRATA_KEYS.items():
        sl = block.get(key)
        if not isinstance(sl, dict):
            continue
        recs = []
        for arm, leaf in sl.items():
            if arm.startswith("_") or not isinstance(leaf, dict) or "/" not in arm:
                continue
            m, u = arm.split("/", 1)
            recs.append({"model": m, "mechanism": u,
                         "coverage_95": leaf.get("coverage_95"),
                         "joint_path_coverage_95": leaf.get("joint_path_coverage_95"),
                         "n_cells": leaf.get("n_cells")})
        if recs:
            out[label] = pd.DataFrame(recs)
    return out or None


def _strata_from_parquet(placebo: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    ok = valid_rows(placebo)
    for s in STRATA_ORDER:
        pops = [p for p, st in PLACEBO_STRATA.items() if st == s]
        sub = ok[ok["pop"].isin(pops)]
        if sub.empty:
            continue
        st = sub.groupby(["model", "mechanism"])[["coverage_95", "joint_path_coverage_95"]].mean()
        st["n_cells"] = sub.groupby(["model", "mechanism"]).size()
        out[s] = st.reset_index()
    return out


def tab_twin_crises(df: pd.DataFrame, sens: dict | None, w: TableWriter):
    placebo, shift = regime_frame(df, "placebo"), regime_frame(df, "shift")
    st_p = cell_stats(valid_rows(placebo), placebo[placebo["error"].notna()],
                      ["coverage_95", "joint_path_coverage_95"]) if not placebo.empty else None
    st_s = cell_stats(valid_rows(shift), shift[shift["error"].notna()],
                      ["coverage_95", "joint_path_coverage_95"]) if not shift.empty else None
    strata = _strata_from_sensitivities(sens)
    strata_src = "sensitivities JSON" if strata else None
    if strata is None and not placebo.empty:
        strata = _strata_from_parquet(placebo)
        strata_src = "placebo parquet"
    strata_cols = [s for s in STRATA_ORDER if strata and s in strata]
    cells = sort_cells([(m, u) for s in (st_p, st_s) if s is not None
                        for m, u in zip(s["model"], s["mechanism"])])
    gp_here = {"placebo": gp_present(placebo), "shift": gp_present(shift)}
    if cells and not any(m == GP_FAMILY for m, _ in cells):
        cells = sort_cells([*cells, (GP_FAMILY, "")])
    colspec = "ll" + ("rrr" if st_p is not None else "c") + ("rrr" if st_s is not None else "c") \
        + ("r" * len(strata_cols) if strata_cols else "c")
    head1 = ["Family", "Mech.",
             "\\multicolumn{3}{c}{placebo 1914--22}" if st_p is not None else "placebo",
             "\\multicolumn{3}{c}{shift 2020--24}" if st_s is not None else "shift",
             f"\\multicolumn{{{len(strata_cols)}}}{{c}}{{placebo cov$_{{95}}$ by stratum}}"
             if strata_cols else "strata"]
    head2 = ["", ""] + (["cov$_{95}$", "joint", "$n$"] if st_p is not None else [PENDING]) \
        + (["cov$_{95}$", "joint", "$n$"] if st_s is not None else [PENDING]) \
        + ([tex(s) for s in strata_cols] if strata_cols else [PENDING])
    body = []
    last = None
    for m, u in cells:
        if last is not None and m != last:
            body.append("\\addlinespace[2pt]")
        first_of_fam = (m != last)
        last = m
        line = [fam(m) if first_of_fam else "", mech(u) if u else "all arms"]
        for st, reg in ((st_p, "placebo"), (st_s, "shift")):
            if st is None:
                line.append("--")
                continue
            if m == GP_FAMILY and not gp_here[reg]:
                line.append(gp_pending_cell(3))
                continue
            row = _lookup(st, m, u) if u else None
            line += ([f3(row["coverage_95"]), f3(row["joint_path_coverage_95"]), fint(row["n"])]
                     if row is not None else ["--", "--", "0"])
        if strata_cols:
            if m == GP_FAMILY and not gp_here["placebo"]:
                line.append(gp_pending_cell(len(strata_cols)))
            else:
                for s in strata_cols:
                    row = _lookup(strata[s], m, u) if u else None
                    line.append(f3(row["coverage_95"]) if row is not None else "--")
        else:
            line.append("--")
        body.append(" & ".join(line) + " \\\\")
    if not cells:
        body.append(pending_row("placebo and shift", _ncols(colspec)))
    notes = [regime_note(df, "placebo"), regime_note(df, "shift"), DESCRIPTIVE_NOTE,
             "Placebo: train $\\le$1913, test 1914--1922 ($h=1\\ldots9$), 11 populations; "
             "shift: train $\\le$2019, test 2020--2024 ($h=1\\ldots5$), 20 populations. "
             "Joint-path coverage spans each regime's full horizon, so the two joint "
             "columns are at different path lengths and are read within regime.",
             ("Strata per PREREGISTRATION-ADDENDUM-1 \\S A (neutral register-based / "
              "belligerent total / civilian-only), source: " + strata_src + ".")
             if strata_cols else
             f"Strata columns (addendum 1 \\S A): {PENDING} -- neither a sensitivities JSON "
             "with a computed \\texttt{strata.placebo} block nor a placebo parquet was supplied.",
             "Every registered family runs in the placebo regime (PREREGISTRATION.md places "
             "no family restriction on it; pass 1 = all families but GP, pass 2 = GP). "
             "Descriptive, no transfer regression.", gp_note(df)]
    w.write("tab-twin-crises", colspec, [" & ".join(head1) + " \\\\", " & ".join(head2) + " \\\\"],
            body, notes)


_NUM_RATIO = re.compile(r"\d+/\d+")


def reason_excerpt(err: str, width: int = 58) -> str:
    s = re.sub(r"^\w+Error:\s*", "", str(err))
    s = _NUM_RATIO.sub("k/N", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > width:
        s = s[: width - 1].rstrip() + "\\ldots"
    return tex(s)


def infeasible_table(df: pd.DataFrame) -> pd.DataFrame:
    """(regime, pop, model, mechanism, class, reason, n) for every error row."""
    err = df[df["error"].notna()]
    if err.empty:
        return pd.DataFrame(columns=["regime_group", "pop", "model", "mechanism", "error_class", "reason", "n"])
    tab = (err.assign(reason=err["error"].map(reason_excerpt))
              .groupby(["regime_group", "pop", "model", "mechanism", "error_class"], sort=False)
              .agg(reason=("reason", "first"), n=("reason", "size")).reset_index())
    cls_rank = {"machine": 0, "structural": 1, "method": 2, "other": 3}
    tab["_c"] = tab["error_class"].map(cls_rank)
    tab["_k"] = [_order_key(m, u) for m, u in zip(tab["model"], tab["mechanism"])]
    return tab.sort_values(["regime_group", "_c", "pop", "_k"]).drop(columns=["_c", "_k"]).reset_index(drop=True)


def infeasible_compact(tab: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-population listing to (regime, model, mechanism,
    class): populations affected (sorted codes), number of populations and
    error rows."""
    if tab.empty:
        return pd.DataFrame(columns=["regime_group", "model", "mechanism", "error_class",
                                     "pops", "n_pops", "n"])
    g = tab.groupby(["regime_group", "model", "mechanism", "error_class"], sort=False)
    out = g.agg(pops=("pop", lambda s: ", ".join(sorted(set(s)))),
                n_pops=("pop", "nunique"), n=("n", "sum")).reset_index()
    cls_rank = {"machine": 0, "structural": 1, "method": 2, "other": 3}
    out["_c"] = out["error_class"].map(cls_rank)
    out["_k"] = [_order_key(m, u) for m, u in zip(out["model"], out["mechanism"])]
    return out.sort_values(["regime_group", "_c", "_k"]).drop(columns=["_c", "_k"]).reset_index(drop=True)


INFEASIBLE_CLASS_NOTE = (
    "Classes follow \\texttt{scripts/final\\_qa.py}: \\emph{structural} = design-floor "
    "cell (panel too short for the mechanism's calibration window, admissibility "
    "floor $n_{\\mathrm{train}}\\ge 15$, addendum 3 \\S4/\\S11) -- a property of the "
    "design, tabulated and dropped by the common-cell restriction; \\emph{method} = "
    "the family's own sampler refused to produce a predictive law (SVAR explosive "
    "coefficient draws rejected, never clipped, addendum 3 \\S7; Poisson "
    "composition overflow) -- a finding about the family; \\emph{machine} = a broken "
    "run, which aborts table generation and is therefore always empty here.")


def tab_infeasible(df: pd.DataFrame, w: TableWriter):
    """Main-text fragment: compact (model, mechanism, class) aggregate with the
    populations affected. The per-population listing is the appendix
    longtable ``tab-infeasible-full``."""
    tab = infeasible_table(df)
    machine = tab[tab["error_class"] == "machine"]
    if len(machine):
        raise SystemExit(
            f"{int(machine['n'].sum())} machine-failure rows in the inputs "
            f"({machine[['pop', 'model']].drop_duplicates().values.tolist()}); "
            "run scripts/final_qa.py and re-run those parts -- tables are NOT generated.")
    compact = infeasible_compact(tab)
    ncols = 6
    body = []
    counts = {}
    for r in EXPECTED_REGIMES:
        sub = compact[compact["regime_group"] == r]
        if regime_frame(df, r).empty:
            body.append(pending_row(r, ncols))
            continue
        body += span_row(f"\\textbf{{{r} regime}}", ncols)
        if not gp_present(regime_frame(df, r)):
            body.append(f"\\multicolumn{{{ncols}}}{{l}}{{{fam(GP_FAMILY)} -- {GP_PENDING}}} \\\\")
        if sub.empty:
            body.append(f"\\multicolumn{{{ncols}}}{{l}}{{no error rows}} \\\\")
        for _, x in sub.iterrows():
            body.append(" & ".join([fam(x["model"]), mech(x["mechanism"]), tex(x["error_class"]),
                                    tex(x["pops"]), fint(x["n_pops"]), fint(x["n"])]) + " \\\\")
        counts[r] = tab[tab["regime_group"] == r].groupby("error_class")["n"].sum().to_dict()
    notes = [regime_note(df, r) for r in EXPECTED_REGIMES]
    notes += [
        INFEASIBLE_CLASS_NOTE,
        "Populations: HMD codes of the populations with at least one error row in the "
        "cell; \\#pop = their number; $n$ = error rows (populations, sexes and origins "
        "pooled). The per-population listing with the error messages is "
        "Table~\\ref{tab:infeasible-full} (appendix).",
        "Totals by class: " + ("; ".join(
            f"{r}: " + ", ".join(f"{c} {int(n)}" for c, n in sorted(cs.items())) if cs else f"{r}: none"
            for r, cs in counts.items()) or PENDING) + ".",
        gp_note(df),
    ]
    w.write("tab-infeasible", "llp{1.5cm}p{5.6cm}rr",
            ["Family & Mech. & Class & Populations affected & \\#pop & $n$ \\\\"],
            body, notes)
    tab_infeasible_full(df, tab, w)


def tab_infeasible_full(df: pd.DataFrame, tab: pd.DataFrame, w: TableWriter):
    """Appendix longtable: every (population, family, mechanism) error cell
    with its class and the first error message."""
    ncols = 6
    body = []
    for r in EXPECTED_REGIMES:
        sub = tab[tab["regime_group"] == r]
        if regime_frame(df, r).empty:
            body.append(pending_row(r, ncols))
            continue
        body += span_row(f"\\textbf{{{r} regime}}", ncols)
        if not gp_present(regime_frame(df, r)):
            body.append(f"\\multicolumn{{{ncols}}}{{l}}{{{fam(GP_FAMILY)} -- {GP_PENDING}}} \\\\")
        if sub.empty:
            body.append(f"\\multicolumn{{{ncols}}}{{l}}{{no error rows}} \\\\")
        for _, x in sub.iterrows():
            body.append(" & ".join([tex(x["pop"]), fam(x["model"]), mech(x["mechanism"]),
                                    tex(x["error_class"]), x["reason"], fint(x["n"])]) + " \\\\")
    notes = [regime_note(df, r) for r in EXPECTED_REGIMES]
    notes += [INFEASIBLE_CLASS_NOTE,
              "Reason: first error message of the group, numerators/denominators of draw "
              "counts abbreviated as k/N; $n$ = error rows (sexes and origins pooled). "
              "Compact aggregate in the main text: Table~\\ref{tab:infeasible}.",
              gp_note(df)]
    w.write_longtable("tab-infeasible-full", "lllp{1.6cm}p{6.2cm}r",
                      ["Pop. & Family & Mech. & Class & Reason (excerpt) & $n$ \\\\"],
                      body, notes,
                      caption="Infeasible cells by population, family and mechanism "
                              "(\\texttt{scripts/final\\_qa.py} classes); full listing "
                              "behind Table~\\ref{tab:infeasible}.",
                      label="tab:infeasible-full")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def build_all(df: pd.DataFrame, analyses: dict[str, dict], sens: dict | None,
              out_dir: str | Path, sources: list[str] | None = None,
              snapshot: bool = True, hmd_deaths: str | Path | None = None,
              hmd_exposures: str | Path | None = None) -> list[Path]:
    """Generate every table into out_dir; returns the written paths.

    Raises SystemExit before writing anything if a machine-failure row is
    present (the QA gate's rule: re-run, never analyse around).
    ``hmd_deaths`` / ``hmd_exposures`` feed tab-populations only (generated
    from the data files); None or a missing file prints pending cells.
    """
    df = prepare_rows(df)
    if (df["error_class"] == "machine").any():
        tab_infeasible(df, TableWriter(Path(out_dir), [], snapshot))  # raises
    w = TableWriter(Path(out_dir), sources or [], snapshot)
    tab_populations(w, hmd_deaths, hmd_exposures)
    tab_h1(df, w)
    tab_h2(df, w)
    tab_h3(df, w)
    tab_h4(df, w)
    tab_h5(df, w)
    tab_murphy(df, w)
    tab_pit(df, w)
    tab_dm_mcs(analyses, w)
    tab_twin_crises(df, sens, w)
    tab_infeasible(df, w)
    return w.written


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", action="append", default=[],
                   help="runner parquet (repeatable): results/<regime>.parquet and the "
                        "second-pass results/<regime>_gp.parquet; rows merge by regime")
    p.add_argument("--analysis", action="append", default=[],
                   help="scripts/analyse.py JSON (repeatable; one per regime)")
    p.add_argument("--sensitivities", default=None,
                   help="optional scripts/sensitivities.py JSON; its strata.placebo key "
                        "feeds the tab-twin-crises stratum columns; absent -> placeholder")
    p.add_argument("--hmd-deaths", default=str(DEFAULT_HMD_DEATHS),
                   help="HMD bulk Deaths_1x1 file (tab-populations is generated from the "
                        "data files); default Dataset/deaths/Deaths_1x1/Deaths_1x1.txt")
    p.add_argument("--hmd-exposures", default=str(DEFAULT_HMD_EXPOSURES),
                   help="HMD bulk Exposures_1x1 file (zero-exposure training cells); "
                        "default Dataset/exposures/Exposures_1x1/Exposures_1x1.txt")
    p.add_argument("--out", default=str(ROOT / "paper" / "tables"))
    p.add_argument("--final", action="store_true",
                   help="suppress the NOT FINAL stamp (refused if any input is a snapshot)")
    args = p.parse_args(argv)

    sens = load_sensitivities(args.sensitivities)
    snapshot = (any(is_snapshot_path(x) for x in args.parquet + args.analysis)
                or bool((sens or {}).get("snapshot", False)))
    if args.final and snapshot:
        p.error("--final given but an input is a snapshot file; refusing to stamp as final")
    df = load_rows(args.parquet)
    analyses = load_analyses(args.analysis)
    sources = [Path(x).name for x in args.parquet + args.analysis]
    if args.sensitivities and sens is not None:
        sources.append(Path(args.sensitivities).name)
    hmd_ok = Path(args.hmd_deaths).exists() and Path(args.hmd_exposures).exists()
    written = build_all(df, analyses, sens, args.out, sources=sources,
                        snapshot=snapshot or not args.final, hmd_deaths=args.hmd_deaths,
                        hmd_exposures=args.hmd_exposures)
    for path in written:
        print(f"[make_tables] wrote {path}")
    gp_regs = [r for r in present_regimes(df) if gp_present(regime_frame(df, r))]
    print(f"[make_tables] regimes present: {present_regimes(df) or 'none'}; "
          f"GP rows in: {gp_regs or 'none (pending second-pass parquet)'}; "
          f"analyses: {sorted(analyses) or 'none'}; sensitivities: "
          f"{'yes' if sens else 'absent (placeholder)'}; "
          f"tab-populations: {'from the data files' if hmd_ok else 'pending (HMD bulk files not found)'}; "
          f"stamp: {'SNAPSHOT - NOT FINAL' if (snapshot or not args.final) else 'final'} "
          "(tab-populations carries no stamp: data-file derived)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
