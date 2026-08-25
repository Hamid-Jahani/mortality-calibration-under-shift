"""Gate 2 verdict: Python PoissonLeeCarter vs StMoMo on the shared panel."""
from pathlib import Path

import numpy as np
import pandas as pd

from mortcal.models import PoissonLeeCarter

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "results" / "parity"

D = pd.read_csv(P / "D_swe_male.csv", index_col=0).to_numpy()
E = pd.read_csv(P / "E_swe_male.csv", index_col=0).to_numpy()
r = pd.read_csv(P / "stmomo_plc_params.csv")
a_r = r[r.param == "alpha"].value.to_numpy()
b_r = r[r.param == "beta"].value.to_numpy()
k_r = r[r.param == "kappa"].value.to_numpy()

m = PoissonLeeCarter(max_iter=200000, tol=0.0).fit(D, E)  # run to cap: loglik-flat directions need it for 1e-6 param parity

ll_py = float((D * np.log(E * np.exp(m.alpha[:, None] + np.outer(m.beta, m.kappa)))
               - E * np.exp(m.alpha[:, None] + np.outer(m.beta, m.kappa))).sum())

def rel(x, y):
    return float(np.max(np.abs(x - y) / np.maximum(np.abs(y), 1e-12)))

print(f"max rel diff  alpha: {rel(m.alpha, a_r):.3e}")
print(f"max rel diff  beta : {rel(m.beta, b_r):.3e}")
print(f"max rel diff  kappa: {rel(m.kappa, k_r):.3e}")
print(f"python loglik (kernel): {ll_py:.2f}")
corr = np.corrcoef(m.kappa, k_r)[0, 1]
print(f"kappa correlation: {corr:.10f}")
