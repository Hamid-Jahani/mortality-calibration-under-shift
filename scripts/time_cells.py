"""Per-cell wall-time probe at production settings (n_samples=1000, B=200).

Writes results/timings.json incrementally so a killed run keeps what it
measured. Read by docs/STATUS.md's sweep-cost estimate.
"""
import json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mortcal.data import build_panel                      # noqa: E402
from mortcal.runner import ADMISSIBLE, run_cell, _pivot_matrices   # noqa: E402
from mortcal.splits import SHIFT                          # noqa: E402

PROBES = [("PLC", "native"), ("PLC", "pboot"), ("PLC", "split_conf"),
          ("PLC", "enbpi"), ("PLC", "copula_conf"),
          ("LC", "native"), ("CBD", "native"), ("RH", "native"), ("SVAR", "native"),
          ("GP", "native"), ("GP", "split_conf"),
          ("NB", "native"), ("NB", "ensemble"), ("NB", "dropout"),
          ("NLC", "ensemble"), ("NLC", "dropout"), ("NLC", "split_conf"),
          ("CNN", "ensemble"), ("CNN", "dropout"),
          ("LSTM", "ensemble"), ("LSTM", "dropout")]


def main():
    DS = ROOT / "Dataset"
    panel = build_panel(DS / "deaths/Deaths_1x1/Deaths_1x1.txt",
                        DS / "exposures/Exposures_1x1/Exposures_1x1.txt",
                        pops=["SWE"], age_max=99)
    sub = panel[(panel["pop"] == "SWE") & (panel["sex"] == "male")]
    D, E, oD, oE = _pivot_matrices(sub, SHIFT.train_max_year, SHIFT.test_years)
    print(f"SWE male: train {D.shape}, test {oD.shape}", flush=True)

    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "timings.json"
    out_path.parent.mkdir(exist_ok=True)
    # optional argv[2]: comma list of "FAM/mech" to probe (subset of PROBES),
    # e.g. NB/native,NLC/ensemble — for A/B timing without the full 20-min list
    probes = PROBES
    if len(sys.argv) > 2:
        want = {tuple(s.split("/")) for s in sys.argv[2].split(",") if s}
        probes = [p for p in PROBES if p in want]
    res = {}
    for (m, u) in probes:
        if (m, u) not in ADMISSIBLE:
            print(f"  {m}/{u}: inadmissible", flush=True)
            continue
        kw = {"B": 200} if u == "pboot" else None
        t0 = time.time()
        try:
            run_cell(D, E, m, u, h=5, n_samples=1000, rng=np.random.default_rng(1),
                     obs_D=oD, obs_E=oE, mech_kwargs=kw)
            dt = time.time() - t0
            res[f"{m}/{u}"] = round(dt, 2)
            print(f"  {m:5s}/{u:12s} {dt:9.1f}s", flush=True)
        except Exception as exc:
            res[f"{m}/{u}"] = f"ERROR {type(exc).__name__}: {exc}"
            print(f"  {m:5s}/{u:12s} ERROR {type(exc).__name__}: {str(exc)[:70]}", flush=True)
        out_path.write_text(json.dumps(res, indent=1))
    print(f"\nwritten -> {out_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
