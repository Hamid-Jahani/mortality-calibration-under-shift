"""The split module is the transcription of PREREGISTRATION.md's "Regimes and
populations" section; these tests pin the transcription to the registered text
(known truth = the document itself) and check the expanding-origin arithmetic
against hand-computed cases, including the leakage invariant of methodology
rule 2 (train years strictly precede test years, always)."""
import dataclasses

import pytest

from mortcal.splits import (
    PLACEBO,
    PLACEBO_POPS,
    REGIMES,
    SHIFT,
    SHIFT_POPS,
    STABLE,
    Regime,
    expanding_origins,
)


# ---------------------------------------------------------------------------
# expanding_origins: hand-computed truth + invariants
# ---------------------------------------------------------------------------

def test_expanding_origins_hand_computed():
    pairs = expanding_origins(2000, 2004, step=2, max_h=3, cap_year=2005)
    assert pairs == [
        (2000, (2001, 2002, 2003)),
        (2002, (2003, 2004, 2005)),
        (2004, (2005,)),                    # cap truncates the last origin
    ]


def test_expanding_origins_drops_empty_and_ignores_cap_none():
    # cap == first origin => every test slice empty => nothing returned
    assert expanding_origins(2000, 2004, step=2, max_h=3, cap_year=2000) == []
    pairs = expanding_origins(2000, 2004, step=4, max_h=2)      # no cap
    assert pairs == [(2000, (2001, 2002)), (2004, (2005, 2006))]
    with pytest.raises(ValueError):
        expanding_origins(2000, 2004, step=0)


def test_expanding_origins_never_leak():
    """Rule 2: every test year strictly follows its origin, in every pair."""
    for origin, test in expanding_origins(1950, 2014, step=1, max_h=7, cap_year=2019):
        assert min(test) == origin + 1
        assert test == tuple(range(origin + 1, origin + 1 + len(test)))


# ---------------------------------------------------------------------------
# SHIFT — primary regime
# ---------------------------------------------------------------------------

def test_shift_regime_matches_preregistration():
    assert SHIFT.train_max_year == 2019
    assert SHIFT.test_years == (2020, 2021, 2022, 2023, 2024)
    assert SHIFT.horizons == (1, 2, 3, 4, 5)
    assert len(SHIFT.pops) == 20
    assert SHIFT.pops == SHIFT_POPS
    for p in ("BEL", "CHE", "CHL", "DNK", "EST", "FIN", "HKG", "HRV", "ISL",
              "JPN", "KOR", "LTU", "LUX", "LVA", "NOR", "PRT", "SVK", "SWE",
              "TWN", "USA"):
        assert p in SHIFT.pops


# ---------------------------------------------------------------------------
# STABLE — control regime
# ---------------------------------------------------------------------------

def test_stable_origins_and_cap():
    assert tuple(r.train_max_year for r in STABLE) == tuple(range(1990, 2015, 2))
    for r in STABLE:
        assert max(r.test_years) <= 2019          # the registered cap
        assert min(r.test_years) == r.train_max_year + 1   # no gap, no leak
        assert r.horizons == tuple(range(1, len(r.test_years) + 1))
        assert len(r.test_years) <= 5
        assert r.pops == SHIFT_POPS               # same 20 populations


# ---------------------------------------------------------------------------
# PLACEBO — WWI + 1918 flu
# ---------------------------------------------------------------------------

def test_placebo_regime_matches_preregistration():
    assert PLACEBO.train_max_year == 1913
    assert PLACEBO.test_years == tuple(range(1914, 1923))
    assert PLACEBO.horizons == tuple(range(1, 10))
    assert PLACEBO.pops == PLACEBO_POPS
    for p in ("BEL", "FRACNP", "GBRCENW"):        # registered exclusions
        assert p not in PLACEBO.pops
    for p in ("CHE", "DNK", "FIN", "FRATNP", "GBRTENW", "GBR_SCO", "ISL",
              "ITA", "NLD", "NOR", "SWE"):
        assert p in PLACEBO.pops


# ---------------------------------------------------------------------------
# Regime container: runner config + immutability
# ---------------------------------------------------------------------------

def test_regimes_mapping_is_the_runner_config():
    assert set(REGIMES) == {"shift", "stable", "placebo"}
    assert REGIMES["shift"] is SHIFT
    assert REGIMES["placebo"] is PLACEBO
    assert REGIMES["stable"] is STABLE


def test_regime_is_frozen():
    """The constants transcribe a pre-registration; mutating one at runtime
    must be an error, not a silent protocol deviation."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        SHIFT.train_max_year = 2020
