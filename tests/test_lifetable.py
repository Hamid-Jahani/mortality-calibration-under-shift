"""Life-table module validated against synthetic truth from KNOWN processes,
then against HMD's own published life tables (validation gate 3 in
PREREGISTRATION.md). Two synthetic oracles:

1. Constant hazard mu — closed form. Because q = m/(1+(1-a)m) is exactly the
   statement m = d/L, every L_x = d_x/mu and the closure gives L_A = l_A/mu,
   so T_0 = l_0/mu EXACTLY, for any a_x. e0 must equal 1/mu to float precision.
2. Gompertz hazard — no closed form; truth comes from numerical integration of
   the continuous survival function on a 0.001-year grid (the same
   numerical-oracle idea as the R StMoMo parity gate). The single-year table is
   a discretisation, so agreement is to a documented tolerance, not exact.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import cumulative_trapezoid

from mortcal.lifetable import annuity_factor, life_expectancy, life_table

N_AGES = 111  # ages 0..109 plus the 110+ open group, HMD layout

LT_PATH = (
    Path(__file__).resolve().parents[1]
    / "Dataset" / "lt_both" / "bltper_1x1" / "bltper_1x1.txt"
)


# ---------------------------------------------------------------------------
# Oracle 1: constant hazard (exact)
# ---------------------------------------------------------------------------

def test_constant_hazard_e0_is_exact():
    """mu constant => m_x = mu at every age and e0 = 1/mu exactly (see module
    docstring for why the identity is exact under the m=d/L conversion)."""
    for mu in (0.01, 0.05, 0.2):
        mx = np.full(N_AGES, mu)
        assert abs(life_expectancy(mx) - 1.0 / mu) < 1e-9


# ---------------------------------------------------------------------------
# Oracle 2: Gompertz hazard vs fine-grid numerical integration
# ---------------------------------------------------------------------------

def _gompertz_truth(B=5e-5, c=1.10, top=120.0, step=1e-3):
    """Exact-to-quadrature survival curve + the single-year m_x it implies.

    Returns (mx[N_AGES], e0_true, e65_true, a65_true) where the truths are
    computed straight from the continuous law: e_x = int S / S(x), central
    rate m_x = (S(x)-S(x+1)) / int_x^{x+1} S (so m_x is the TRUE central
    death rate of the process, not an approximation)."""
    x = np.arange(0.0, top + step / 2, step)
    S = np.exp(-cumulative_trapezoid(B * c ** x, x, initial=0.0))
    per = round(1.0 / step)
    e0 = np.trapezoid(S, x)
    e65 = np.trapezoid(S[65 * per:], x[65 * per:]) / S[65 * per]
    mx = np.array([
        (S[a * per] - S[(a + 1) * per])
        / np.trapezoid(S[a * per:(a + 1) * per + 1], dx=step)
        for a in range(N_AGES)
    ])
    S_int = S[::per]                       # survival at integer ages
    t = np.arange(int(top) - 65)
    a65 = float(np.sum(1.02 ** -t * S_int[65 + t] / S_int[65]))
    return mx, float(e0), float(e65), a65


def test_gompertz_life_table_matches_numerical_oracle():
    """Feeding the TRUE Gompertz central rates through our table must recover
    the continuous-law e0/e65 up to the a_x=0.5 discretisation error, which for
    a smooth hazard is O(1e-3) years — tolerance 0.01 (dry-run: 1.3e-3)."""
    mx, e0_true, e65_true, _ = _gompertz_truth()
    lt = life_table(mx)
    assert abs(lt["ex"][0] - e0_true) < 0.01
    assert abs(lt["ex"][65] - e65_true) < 0.01


def test_gompertz_annuity_matches_numerical_oracle():
    """ä65 at 2% from our period table vs the same sum taken on the exact
    integer-age survival curve. Tolerance 0.01 (dry-run error: 1.1e-3)."""
    mx, _, _, a65_true = _gompertz_truth()
    assert abs(annuity_factor(mx, x0=65, i=0.02) - a65_true) < 0.01


# ---------------------------------------------------------------------------
# Interface contract: vectorisation, sanity, infant rule
# ---------------------------------------------------------------------------

def test_vectorised_matches_per_row_and_shapes():
    """[n, n_ages] input -> [n] outputs identical to n separate 1-D calls."""
    base, _, _, _ = _gompertz_truth()
    scales = np.array([0.6, 1.0, 1.8])
    mx = scales[:, None] * base[None, :]                  # [3, N_AGES]
    e0 = life_expectancy(mx)
    e65 = life_expectancy(mx, age=65)
    a65 = annuity_factor(mx)
    assert e0.shape == e65.shape == a65.shape == (3,)
    for j in range(3):
        assert e0[j] == pytest.approx(life_expectancy(mx[j]))
        assert a65[j] == pytest.approx(annuity_factor(mx[j]))
    # heavier mortality => shorter life, cheaper annuity — strictly monotone
    assert np.all(np.diff(e0) < 0) and np.all(np.diff(a65) < 0)
    lt = life_table(mx)
    assert lt["ex"].shape == (3, N_AGES)


def test_life_table_columns_are_internally_consistent():
    mx, _, _, _ = _gompertz_truth()
    lt = life_table(mx, radix=1e5)
    assert lt["lx"][0] == 1e5
    assert np.all(np.diff(lt["lx"]) <= 0)                        # non-increasing
    np.testing.assert_allclose(lt["dx"][:-1], -np.diff(lt["lx"]))
    np.testing.assert_allclose(lt["Tx"][:-1] - lt["Tx"][1:], lt["Lx"][:-1])
    assert lt["qx"][-1] == 1.0                                    # open group
    assert lt["ex"][-1] == pytest.approx(1.0 / mx[-1])            # e_110 = 1/m


def test_infant_rule_caps():
    """a0 = 0.07 + 1.7*m0 clipped to [0.01, 0.35]."""
    mid = np.full(N_AGES, 0.05)
    assert life_table(mid)["ax"][0] == pytest.approx(0.07 + 1.7 * 0.05)
    hi = np.full(N_AGES, 0.05); hi[0] = 0.5
    assert life_table(hi)["ax"][0] == 0.35
    lo = np.full(N_AGES, 0.05); lo[0] = 1e-9
    assert life_table(lo)["ax"][0] == pytest.approx(0.07)
    assert np.all(life_table(mid)["ax"][1:] == 0.5)


# ---------------------------------------------------------------------------
# Validation gate 3: HMD parity (skipped when the Dataset tree is absent)
# ---------------------------------------------------------------------------

_LT_COLS = ["pop", "year", "age", "mx", "qx", "ax", "lx", "dx", "Lx", "Tx", "ex"]


def _read_hmd_lifetable(path):
    """Bulk HMD life-table reader, same conventions as mortcal.data.hmd:
    3 preamble lines, whitespace-separated, '.' missing, '110+' -> 110."""
    df = pd.read_csv(path, sep=r"\s+", skiprows=3, names=_LT_COLS,
                     na_values=".", dtype={"pop": str, "age": str})
    df = df[df["pop"] != "PopName"]                # vintage preamble guard
    df["age"] = df["age"].str.replace("+", "", regex=False).astype(np.int16)
    df["year"] = df["year"].astype(np.int16)
    for c in _LT_COLS[3:]:
        df[c] = df[c].astype(np.float64)
    return df


@pytest.mark.skipif(not LT_PATH.exists(), reason="HMD Dataset tree not present")
@pytest.mark.parametrize("pop,year", [("USA", 2019), ("JPN", 2015)])
def test_hmd_parity_e0_e65(pop, year):
    """Our table on HMD's published m_x must reproduce HMD's published e0 and
    e65 within 0.15 years. The tolerance covers the documented a_x
    simplifications (HMD uses Andreev-Kingkade a0 and its own old-age
    treatment); observed disagreement is <0.01 years."""
    df = _read_hmd_lifetable(LT_PATH)
    sub = df[(df["pop"] == pop) & (df["year"] == year)].sort_values("age")
    assert list(sub["age"]) == list(range(N_AGES)), f"{pop} {year} ages incomplete"
    lt = life_table(sub["mx"].to_numpy())
    hmd_ex = sub["ex"].to_numpy()
    assert abs(lt["ex"][0] - hmd_ex[0]) < 0.15, f"e0 {lt['ex'][0]:.3f} vs {hmd_ex[0]}"
    assert abs(lt["ex"][65] - hmd_ex[65]) < 0.15, f"e65 {lt['ex'][65]:.3f} vs {hmd_ex[65]}"
    # same conversion at closed adult ages => qx agrees to publication rounding
    q_diff = np.abs(lt["qx"][1:100] - sub["qx"].to_numpy()[1:100])
    assert q_diff.max() < 5e-5, f"qx diverges at age {1 + int(q_diff.argmax())}"
