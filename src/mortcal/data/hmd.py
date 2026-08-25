"""Readers for HMD bulk 1x1 text files.

Format (all-population concatenated bulk files, HMD zipped-data download):

    Deaths (period 1x1),   Last modified: 15 Jun 2026; ...
    <blank>
    PopName    Year          Age             Female            Male           Total
    AUS        1921           0              3842.31         5124.54         8966.85
    ...

Ages run 0..109 then "110+" (open age group). Missing values are ".".
Deaths are non-integer because HMD splits deaths across Lexis triangles.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_COLS = ["pop", "year", "age", "female", "male", "total"]


def read_bulk_1x1(path: str | Path, pops: list[str] | None = None) -> pd.DataFrame:
    """Read one HMD bulk 1x1 file (Deaths, Exposures or Mx) into tidy form.

    Returns columns: pop (str), year (int16-compatible int), age (int, 110+ -> 110),
    female / male / total (float64, NaN where HMD reports '.').
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=3,  # title line, blank line, column-header line
        names=_COLS,
        na_values=".",
        dtype={"pop": str, "age": str},
    )
    # guard against vintage variations in preamble length
    df = df[df["pop"] != "PopName"]
    df["age"] = df["age"].str.replace("+", "", regex=False).astype(np.int16)
    df["year"] = df["year"].astype(np.int16)
    for c in ("female", "male", "total"):
        df[c] = df[c].astype(np.float64)
    if pops is not None:
        df = df[df["pop"].isin(pops)].reset_index(drop=True)
    return df


def build_panel(
    deaths_path: str | Path,
    exposures_path: str | Path,
    pops: list[str] | None = None,
    age_max: int = 99,
) -> pd.DataFrame:
    """Join deaths and exposures into the modelling panel.

    One row per (pop, year, age, sex) with deaths D and exposure E.
    Ages above `age_max` are dropped (HMD old-age estimates are model-smoothed
    and the 110+ open group needs special treatment; the study models 0..99).
    Rows with E <= 0 or missing are dropped — Poisson offset log(E) undefined.
    """
    d = read_bulk_1x1(deaths_path, pops)
    e = read_bulk_1x1(exposures_path, pops)
    long_d = d.melt(["pop", "year", "age"], ["female", "male"], "sex", "D")
    long_e = e.melt(["pop", "year", "age"], ["female", "male"], "sex", "E")
    panel = long_d.merge(long_e, on=["pop", "year", "age", "sex"], how="inner")
    panel = panel[panel["age"] <= age_max]
    panel = panel.dropna(subset=["D", "E"])
    panel = panel[panel["E"] > 0].reset_index(drop=True)
    panel["mx"] = panel["D"] / panel["E"]
    return panel
