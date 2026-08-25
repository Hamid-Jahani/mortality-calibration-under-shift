"""Period life tables and annuity factors from central death rates.

Converts an m_x vector (single-year ages 0..A, last age treated as an open
group) into a full period life table and the derived actuarial quantities the
study scores (H5): life expectancy e_x and the whole-life annuity-due factor.

Conventions (documented deviations from the HMD Methods Protocol v6, Wilmoth
et al. 2017, are deliberate and covered by validation gate 3's 0.15-year
tolerance in tests/test_lifetable.py):

* q_x = m_x / (1 + (1 - a_x) m_x)  — the standard central-rate-to-probability
  conversion (HMD Methods Protocol v6, eq. for period tables).
* a_0 = 0.07 + 1.7 m_0, capped to [0.01, 0.35] — the simple infant-separation
  rule (Keyfitz-style linear approximation in the spirit of Andreev & Kingkade
  2015, Demographic Research 33; HMD itself uses the piecewise Andreev-Kingkade
  coefficients, hence the tolerance in the parity gate).
* a_x = 0.5 for all other closed ages (deaths mid-interval on average).
* Open/last age group: q_A = 1, L_A = l_A / m_A, hence e_A = 1/m_A — the
  constant-hazard closure the HMD uses for 110+.
* Everything is vectorised over a leading sample dimension: input [n, n_ages]
  yields tables [n, n_ages] and scalars [n]; 1-D input yields 1-D/scalar output.
  This is how predictive m_x samples ([n, h, n_ages] reshaped per horizon)
  propagate into e_x and annuity intervals through ONE code path (rule 4).
"""
from __future__ import annotations

import numpy as np

_MIN_MX = 1e-12  # guards log/division; far below any observable death rate


def _as_2d(mx: np.ndarray) -> tuple[np.ndarray, bool]:
    """Promote [n_ages] -> [1, n_ages]; return (array, was_1d)."""
    mx = np.asarray(mx, dtype=float)
    if mx.ndim == 1:
        return mx[None, :], True
    if mx.ndim != 2:
        raise ValueError(f"mx must be [n_ages] or [n, n_ages], got ndim={mx.ndim}")
    return mx, False


def life_table(mx: np.ndarray, radix: float = 1.0) -> dict[str, np.ndarray]:
    """Period life table from central death rates.

    Parameters
    ----------
    mx : [n_ages] or [n, n_ages]
        Central death rates for single-year ages 0..A; the last entry is the
        open age group. Values are clipped below at 1e-12.
    radix : float
        l_0 (default 1.0; use 1e5 for HMD-style presentation).

    Returns
    -------
    dict with keys qx, ax, lx, dx, Lx, Tx, ex — each shaped like `mx`.
    """
    m, was_1d = _as_2d(mx)
    m = np.clip(m, _MIN_MX, None)

    ax = np.full_like(m, 0.5)
    ax[:, 0] = np.clip(0.07 + 1.7 * m[:, 0], 0.01, 0.35)  # infant rule, documented above

    qx = m / (1.0 + (1.0 - ax) * m)
    qx = np.clip(qx, 0.0, 1.0)
    qx[:, -1] = 1.0                                       # open group absorbs

    lx = np.empty_like(m)
    lx[:, 0] = radix
    lx[:, 1:] = radix * np.cumprod(1.0 - qx[:, :-1], axis=1)
    dx = lx * qx

    Lx = lx - (1.0 - ax) * dx                             # L_x = l_{x+1} + a_x d_x
    Lx[:, -1] = lx[:, -1] / m[:, -1]                      # constant-hazard closure

    Tx = np.cumsum(Lx[:, ::-1], axis=1)[:, ::-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ex = np.where(lx > 0.0, Tx / lx, 0.0)

    out = {"qx": qx, "ax": ax, "lx": lx, "dx": dx, "Lx": Lx, "Tx": Tx, "ex": ex}
    if was_1d:
        out = {k: v[0] for k, v in out.items()}
    return out


def life_expectancy(mx: np.ndarray, age: int = 0) -> np.ndarray | float:
    """Period life expectancy e_age from central death rates.

    Vectorised: mx [n, n_ages] -> [n]; mx [n_ages] -> float.
    """
    m, was_1d = _as_2d(mx)
    if not 0 <= age < m.shape[1]:
        raise ValueError(f"age {age} outside table 0..{m.shape[1] - 1}")
    ex = life_table(m)["ex"][:, age]
    return float(ex[0]) if was_1d else ex


def annuity_factor(mx: np.ndarray, x0: int = 65, i: float = 0.02) -> np.ndarray | float:
    """Whole-life annuity-due factor ä_x0 = sum_{t>=0} v^t · tP_x0, annual.

    Standard life-contingency definition (e.g. Dickson, Hardy & Waters 2020,
    ch. 5), computed from the PERIOD life table treated as static: tP_x0 =
    l_{x0+t}/l_{x0} with l from `life_table(mx)`, i.e. the current period's
    mortality is assumed to apply to the cohort forever (no further improvement
    inside the factor — the forecast uncertainty enters through the m_x samples,
    not through cohort projection inside this function). The sum truncates at
    the table's top age: survivorship beyond the open group's single row is
    ignored (negligible at x0=65 under the constant-hazard closure). v = 1/(1+i).

    Vectorised: mx [n, n_ages] -> [n]; mx [n_ages] -> float.
    """
    m, was_1d = _as_2d(mx)
    if not 0 <= x0 < m.shape[1]:
        raise ValueError(f"x0 {x0} outside table 0..{m.shape[1] - 1}")
    lx = life_table(m)["lx"]
    l0 = lx[:, x0:x0 + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        tpx = np.where(l0 > 0.0, lx[:, x0:] / l0, 0.0)    # [n, A - x0 + 1]
    v_t = (1.0 + i) ** -np.arange(tpx.shape[1])
    a = (tpx * v_t[None, :]).sum(axis=1)
    return float(a[0]) if was_1d else a
