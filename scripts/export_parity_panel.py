"""Gate 2 (PREREGISTRATION.md): export one shared HMD subset for the
Python-vs-StMoMo oracle parity check. SWE males, 1950-2000, ages 0-89
(all-positive exposures; SWE zero-exposure cells sit at ages >= 90)."""
from pathlib import Path

import numpy as np

from mortcal.data import read_bulk_1x1

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "parity"

d = read_bulk_1x1(ROOT / "Dataset/deaths/Deaths_1x1/Deaths_1x1.txt", pops=["SWE"])
e = read_bulk_1x1(ROOT / "Dataset/exposures/Exposures_1x1/Exposures_1x1.txt", pops=["SWE"])

sel = lambda df: (df[(df.year >= 1950) & (df.year <= 2000) & (df.age <= 89)]
                  .pivot(index="age", columns="year", values="male")
                  .sort_index())
D, E = sel(d), sel(e)
assert D.shape == (90, 51) and (E.to_numpy() > 0).all()
OUT.mkdir(parents=True, exist_ok=True)
D.to_csv(OUT / "D_swe_male.csv")
E.to_csv(OUT / "E_swe_male.csv")
print("exported", D.shape, "deaths sum", float(np.round(D.to_numpy().sum())))
