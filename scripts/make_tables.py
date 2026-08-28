"""Paper tables: runner parquet(s) + analysis JSON(s) -> paper/tables/*.tex.

    python scripts/make_tables.py --parquet results/shift.parquet \
        [--parquet results/placebo.parquet ...] \
        --analysis results/shift_analysis.json [--analysis ...] \
        [--sensitivities results/sensitivities.json] \
        [--hmd-deaths data/Deaths_1x1.txt] --out paper/tables

Consumes what ``scripts/final_qa.py`` has already gated and what
``scripts/analyse.py`` has already tested; refits nothing, re-tests nothing.
Every file is a ``booktabs`` body wrapped in ``threeparttable`` so that
``\\inputtable{<name>}`` (paper/main.tex) drops it straight into a floating
``table`` environment. ``tab-grid.tex`` is static and is never written here.

Scoring discipline enforced in code, not by convention
------------------------------------------------------
* Rows with ``scores_secondary`` (the conformal mechanisms) carry PLACEHOLDER
  CRPS / log score / PIT (uniform-in-interval samples; addendum 2 §3). They
  never enter a proper-score ranking, a PIT table or a PIT-Murphy column;
  they are compared on ``winkler_95`` / ``coverage_95`` only and their
  50 % / 80 % columns are printed as n/a (NaN by design in the runner).
* Arms with a different ``n_ages_scored`` are never averaged into the same
  ranking: CBD (ages 55-99) gets its own block with the support stated.
* Error rows enter no mean. They are counted per cell (``n_err``) and
  tabulated by QA class in ``tab-infeasible`` using the regexes of
  ``scripts/final_qa.py``; a machine-failure row aborts table generation.
* A ranking (tab-h1) is a contrast, so it is computed on the common-cell
  intersection of addendum 3 §11 and reports kept / dropped units. The
  descriptive family x mechanism tables are uncensored and state so.
* Absent regimes (no parquet supplied) are printed as explicit ``pending``
  placeholders, never dropped silently. Snapshot inputs stamp every file
  with a NOT FINAL first line.
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

TABLE_NAMES = (
    "tab-populations", "tab-h1-rankings", "tab-h2-coverage", "tab-h3-joint",
    "tab-h4-age", "tab-h5-actuarial", "tab-murphy", "tab-pit", "tab-dm-mcs",
    "tab-twin-crises", "tab-infeasible",
)

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
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        df["_source"] = Path(p).name
        frames.append(df)
    if not frames:
        return prepare_rows(pd.DataFrame(columns=UNIT_KEYS + ["model", "mechanism", "error"]))
    return prepare_rows(pd.concat(frames, ignore_index=True))


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

    def header(self) -> str:
        lines = []
        if self.snapshot:
            regs = ", ".join(f"results/{r}.parquet" for r in EXPECTED_REGIMES)
            lines.append(f"% GENERATED SNAPSHOT - NOT FINAL - regenerate from {regs}")
        lines.append("% GENERATED by scripts/make_tables.py on "
                     f"{_dt.date.today().isoformat()} from "
                     f"{', '.join(self.sources) or '(no inputs)'}; do not hand-edit.")
        return "\n".join(lines)

    def write(self, name: str, colspec: str, header_rows: list[str],
              body_rows: list[str], notes: list[str], size: str = "\\small",
              tabcolsep: str = "4pt") -> Path:
        parts = [self.header(),
                 f"\\begin{{threeparttable}}{size}\\setlength{{\\tabcolsep}}{{{tabcolsep}}}",
                 f"\\begin{{tabular}}{{@{{}}{colspec}@{{}}}}",
                 "\\toprule",
                 *header_rows,
                 "\\midrule",
                 *(body_rows or [f"\\multicolumn{{{_ncols(colspec)}}}{{l}}{{{PENDING}: no rows}} \\\\"]),
                 "\\bottomrule",
                 "\\end{tabular}",
                 "\\begin{tablenotes}\\footnotesize"]
        parts += [f"\\item {n}" for n in notes]
        parts += ["\\end{tablenotes}", "\\end{threeparttable}", ""]
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
        line = [fam(m) if m != last_fam else "", mech(u)]
        last_fam = m
        for r in regimes:
            s = stats[r]
            if s is None:
                line.append("--")
                continue
            row = _lookup(s, m, u)
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

def hmd_years(path: str | None) -> dict[str, tuple[int, int]]:
    """pop -> (first data year, training years to the shift origin) from an
    HMD bulk Deaths_1x1 file, using the runner's contiguity rule (addendum 3
    §2: maximal contiguous block of complete years ending at the origin).
    Empty when no file is given."""
    if not path:
        return {}
    from mortcal.data.hmd import read_bulk_1x1
    d = read_bulk_1x1(path, list(SHIFT_POPS))
    d = d[d["age"] <= 99]
    out = {}
    for p, g in d.groupby("pop"):
        complete = g.groupby("year")[["female", "male"]].apply(lambda x: x.notna().all().all())
        years = sorted(int(y) for y, ok in complete.items() if ok)
        first = years[0] if years else None
        train = [y for y in years if y <= SHIFT.train_max_year]
        gaps = np.flatnonzero(np.diff(np.asarray(train)) > 1)
        if len(gaps):
            train = train[gaps[-1] + 1:]
        out[p] = (first, len(train))
    return out


def tab_populations(df: pd.DataFrame, w: TableWriter, hmd_deaths: str | None):
    shift = regime_frame(df, "shift")
    pops = list(SHIFT_POPS) if shift.empty else sorted(set(SHIFT_POPS) | set(shift["pop"]))
    years = hmd_years(hmd_deaths)
    body = []
    for p in pops:
        sub = shift[shift["pop"] == p]
        fy, ntr = years.get(p, (None, None))
        zero = {}
        for sex in ("female", "male"):
            s = sub[sub["sex"] == sex]["n_zero_death_cells"] if not sub.empty else pd.Series(dtype=float)
            zero[sex] = s.max() if len(s.dropna()) else np.nan
        strat = PLACEBO_STRATA.get(p, "none")
        elig = "yes" if p in PLACEBO_POPS else "no"
        body.append(" & ".join([
            tex(p),
            fint(fy) if fy is not None else PENDING,
            fint(ntr) if ntr is not None else PENDING,
            fint(zero["female"]) if not sub.empty else PENDING,
            fint(zero["male"]) if not sub.empty else PENDING,
            elig, strat]) + " \\\\")
    placebo_only = [p for p in PLACEBO_POPS if p not in pops]
    if placebo_only:
        body.append("\\midrule")
        body += span_row("Placebo-only populations (train $\\le$1913, test 1914--1922)", 7, rule=False)
        for p in placebo_only:
            body.append(" & ".join([tex(p), "--", "--", "--", "--", "yes",
                                    PLACEBO_STRATA.get(p, "none")]) + " \\\\")
    notes = [
        regime_note(df, "shift"),
        "First data year and training years: maximal contiguous block of complete "
        "years ending at the 2019 origin (addendum 3 \\S2; BEL trains 1919--2019)"
        + (", from the HMD bulk file supplied via \\texttt{--hmd-deaths}." if years
           else f"; {PENDING} until an HMD \\texttt{{Deaths\\_1x1}} file is supplied."),
        "Zero-death cells: test-window 2020--2024 cells with $D<0.5$ on ages 0--99 "
        "before any model age mask (addendum 3 \\S10), maximum over the population's "
        "rows, by sex.",
        "Placebo eligibility and stratum per PREREGISTRATION-ADDENDUM-1 \\S A: "
        "neutral / register-based (CHE, DNK, FIN, ISL, NLD, NOR, SWE), belligerent "
        "total series (FRATNP, GBRTENW, ITA), civilian-only (GBR\\_SCO); BEL excluded "
        "(missing 1914--1918).",
    ]
    w.write("tab-populations", "lrrrrll",
            ["Pop. & First year & Train yrs $\\le$2019 & \\multicolumn{2}{c}{Zero-death cells 2020--24} & Placebo & Stratum \\\\",
             " & & & F & M & & \\\\"],
            body, notes)


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
    ]
    w.write("tab-h1-rankings", "llrrrrrr",
            ["Family & Mech. & RMSE & rk & CRPS & rk & log score & rk \\\\"],
            body, notes)


def tab_h2(df: pd.DataFrame, w: TableWriter):
    cols = ["coverage_50", "coverage_80", "coverage_95", "winkler_95"]
    colspec, head, body = block_table(df, cols, ["cov$_{50}$", "cov$_{80}$", "cov$_{95}$", "$\\IS_{95}$"], fmt_levels)
    notes = [regime_note(df, r) for r in EXPECTED_REGIMES]
    notes += [DESCRIPTIVE_NOTE, CONFORMAL_NOTE, CBD_NOTE,
              "$\\IS_{95}$: mean Winkler / interval score of the 95\\% interval on log rates (negatively oriented)."]
    w.write("tab-h2-coverage", colspec, head, body, notes, tabcolsep="3pt")


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
              "only mechanism that constructs a joint band.", CBD_NOTE]
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
              "by Dowd et al.; a reversal is reported as informative (addendum 3 \\S8)."]
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
        for q in ("e0", "e65", "ann65"):
            lo, hi, ob, er = (sub.get(f"{q}_{s}", pd.Series(dtype=float)) for s in ("q025", "q975", "obs", "error"))
            fin = lo.notna() & hi.notna() & ob.notna()
            rec[f"{q}_cov"] = float(((lo[fin] <= ob[fin]) & (ob[fin] <= hi[fin])).mean()) if fin.any() else np.nan
            rec[f"{q}_err"] = float(er.mean()) if er.notna().any() else np.nan
            rec[f"{q}_n"] = int(fin.sum())
        recs.append(rec)
    return pd.DataFrame(recs)


def tab_h5(df: pd.DataFrame, w: TableWriter):
    cols = ["e0_cov", "e0_err", "e65_cov", "e65_err", "ann65_cov", "ann65_err"]
    heads = ["$e_0$ cov", "err", "$e_{65}$ cov", "err", "$\\annuity$ cov", "err"]
    colspec, head, body = block_table(df, cols, heads,
                                      lambda row, m, u, c: f3(row[c], 2 if c.endswith("_err") else 3),
                                      per_block_stats=_h5_stats)
    notes = [regime_note(df, r) for r in EXPECTED_REGIMES]
    notes += [DESCRIPTIVE_NOTE,
              "cov = share of rows whose realised $e_0$, $e_{65}$ or $\\annuity$ (2\\%) "
              "lies inside the model's [2.5\\%, 97.5\\%] sample quantiles; err = mean "
              "(point $-$ observed), years for $e_x$, annuity units for $\\annuity$. "
              "Derived quantities are integrated from the LATENT predictive paths on "
              "the maximal contiguous scored age block from age 0 (addendum 3 \\S3), "
              "so conformal rows are included and unflagged.",
              "CBD's block starts at age 55: $e_0$ is undefined (--) and $e_{65}$ / "
              "$\\annuity$ come from a 55--99 table. Rows whose bounds are missing "
              "are excluded from that quantity's share only."]
    w.write("tab-h5-actuarial", colspec, head, body, notes, tabcolsep="3pt")


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
              "(distributional rows only; n/a for conformal rows).", CBD_NOTE]
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
              CBD_NOTE]
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
    w.write("tab-dm-mcs", "llp{3.2cm}p{5.2cm}rr",
            ["Contrast & Loss & In set / statistic & $p$-values & kept & dropped \\\\"],
            body, notes, size="\\footnotesize", tabcolsep="3pt")


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
        line = [fam(m) if first_of_fam else "", mech(u)]
        for st in (st_p, st_s):
            if st is None:
                line.append("--")
                continue
            row = _lookup(st, m, u)
            line += ([f3(row["coverage_95"]), f3(row["joint_path_coverage_95"]), fint(row["n"])]
                     if row is not None else ["--", "--", "0"])
        if strata_cols:
            for s in strata_cols:
                row = _lookup(strata[s], m, u)
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
             "Classical families only are expected in the placebo regime (neural "
             "families have no registered placebo arm); descriptive, no transfer "
             "regression."]
    w.write("tab-twin-crises", colspec, [" & ".join(head1) + " \\\\", " & ".join(head2) + " \\\\"],
            body, notes, tabcolsep="3pt")


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


def tab_infeasible(df: pd.DataFrame, w: TableWriter):
    tab = infeasible_table(df)
    machine = tab[tab["error_class"] == "machine"]
    if len(machine):
        raise SystemExit(
            f"{int(machine['n'].sum())} machine-failure rows in the inputs "
            f"({machine[['pop', 'model']].drop_duplicates().values.tolist()}); "
            "run scripts/final_qa.py and re-run those parts -- tables are NOT generated.")
    ncols = 6
    body = []
    counts = {}
    for r in EXPECTED_REGIMES:
        sub = tab[tab["regime_group"] == r]
        if regime_frame(df, r).empty:
            body.append(pending_row(r, ncols))
            continue
        body += span_row(f"\\textbf{{{r} regime}}", ncols)
        if sub.empty:
            body.append(f"\\multicolumn{{{ncols}}}{{l}}{{no error rows}} \\\\")
        for _, x in sub.iterrows():
            body.append(" & ".join([tex(x["pop"]), fam(x["model"]), mech(x["mechanism"]),
                                    tex(x["error_class"]), x["reason"], fint(x["n"])]) + " \\\\")
        counts[r] = sub.groupby("error_class")["n"].sum().to_dict()
    notes = [regime_note(df, r) for r in EXPECTED_REGIMES]
    notes += [
        "Classes follow \\texttt{scripts/final\\_qa.py}: \\emph{structural} = design-floor "
        "cell (panel too short for the mechanism's calibration window, admissibility "
        "floor $n_{\\mathrm{train}}\\ge 15$, addendum 3 \\S4/\\S11) -- a property of the "
        "design, tabulated and dropped by the common-cell restriction; \\emph{method} = "
        "the family's own sampler refused to produce a predictive law (SVAR explosive "
        "coefficient draws rejected, never clipped, addendum 3 \\S7; Poisson "
        "composition overflow) -- a finding about the family; \\emph{machine} = a broken "
        "run, which aborts table generation and is therefore always empty here.",
        "Reason: first error message of the group, numerators/denominators of draw "
        "counts abbreviated as k/N; $n$ = error rows (sexes and origins pooled).",
        "Totals by class: " + ("; ".join(
            f"{r}: " + ", ".join(f"{c} {int(n)}" for c, n in sorted(cs.items())) if cs else f"{r}: none"
            for r, cs in counts.items()) or PENDING) + ".",
    ]
    w.write("tab-infeasible", "lllp{1.6cm}p{6.2cm}r",
            ["Pop. & Family & Mech. & Class & Reason (excerpt) & $n$ \\\\"],
            body, notes, size="\\footnotesize", tabcolsep="3pt")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def build_all(df: pd.DataFrame, analyses: dict[str, dict], sens: dict | None,
              out_dir: str | Path, sources: list[str] | None = None,
              snapshot: bool = True, hmd_deaths: str | None = None) -> list[Path]:
    """Generate every table into out_dir; returns the written paths.

    Raises SystemExit before writing anything if a machine-failure row is
    present (the QA gate's rule: re-run, never analyse around).
    """
    df = prepare_rows(df)
    if (df["error_class"] == "machine").any():
        tab_infeasible(df, TableWriter(Path(out_dir), [], snapshot))  # raises
    w = TableWriter(Path(out_dir), sources or [], snapshot)
    tab_populations(df, w, hmd_deaths)
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
                   help="runner parquet (repeatable; one per regime)")
    p.add_argument("--analysis", action="append", default=[],
                   help="scripts/analyse.py JSON (repeatable; one per regime)")
    p.add_argument("--sensitivities", default=None,
                   help="optional sibling JSON with placebo_strata; absent -> placeholder")
    p.add_argument("--hmd-deaths", default=None,
                   help="HMD bulk Deaths_1x1 file for first-year / training-years columns")
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
    written = build_all(df, analyses, sens, args.out, sources=sources,
                        snapshot=snapshot or not args.final, hmd_deaths=args.hmd_deaths)
    for path in written:
        print(f"[make_tables] wrote {path}")
    print(f"[make_tables] regimes present: {present_regimes(df) or 'none'}; "
          f"analyses: {sorted(analyses) or 'none'}; sensitivities: "
          f"{'yes' if sens else 'absent (placeholder)'}; "
          f"stamp: {'SNAPSHOT - NOT FINAL' if (snapshot or not args.final) else 'final'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
