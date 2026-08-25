"""Pre-registered evaluation regimes and expanding-origin splits.

Pure year arithmetic plus the fixed population lists — NO data loading here.
Everything in this module transcribes the "Regimes and populations" section of
PREREGISTRATION.md (registered 2026-08-25, before any model was fit); changing
a constant here is a reportable protocol deviation, not a refactor.

Expanding-origin (rolling-origin) evaluation follows Tashman (2000, IJF 16):
the training window always grows from each population's own start year up to
the origin T; test years are T+1..T+h. Never a random split — mortality panels
leak trivially under random splitting (methodology rule 2).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Regime:
    """One pre-registered evaluation regime.

    train_max_year — origin T: training uses ALL years <= T available for a
        population (expanding window; the left edge is each population's own
        start year, which is data-dependent and deliberately not encoded here).
    test_years — the years scored, in order (test_years[j] is horizon j+1).
    horizons — forecast horizons h aligned 1:1 with test_years.
    pops — HMD population codes the regime evaluates.
    """

    name: str
    train_max_year: int
    test_years: tuple[int, ...]
    horizons: tuple[int, ...]
    pops: tuple[str, ...]


def expanding_origins(
    first_origin: int,
    last_origin: int,
    step: int = 2,
    max_h: int = 5,
    cap_year: int | None = None,
) -> list[tuple[int, tuple[int, ...]]]:
    """(train_slice, test_slice) pairs for an expanding-origin design.

    For each origin T in first_origin, first_origin+step, ..., last_origin:
    train_slice is T itself (train = all years <= T, expanding), test_slice is
    the tuple (T+1, ..., T+max_h) truncated at cap_year. Origins whose test
    slice would be empty are dropped. Pure year arithmetic — no data touched.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    out: list[tuple[int, tuple[int, ...]]] = []
    for origin in range(first_origin, last_origin + 1, step):
        test = tuple(
            y for y in range(origin + 1, origin + max_h + 1)
            if cap_year is None or y <= cap_year
        )
        if test:
            out.append((origin, test))
    return out


# ---------------------------------------------------------------------------
# Population lists — fixed in PREREGISTRATION.md, "Regimes and populations".
# ---------------------------------------------------------------------------

#: The 20 populations whose final annual data reach 2024 in the pinned vintage.
SHIFT_POPS: tuple[str, ...] = (
    "BEL", "CHE", "CHL", "DNK", "EST", "FIN", "HKG", "HRV", "ISL", "JPN",
    "KOR", "LTU", "LUX", "LVA", "NOR", "PRT", "SVK", "SWE", "TWN", "USA",
)

#: Continuous 1908-1922 coverage, start <= 1900. BEL excluded (missing
#: occupation-era rows 1914-1918); FRACNP and GBRCENW excluded as overlapping
#: variants of FRATNP and GBRTENW.
PLACEBO_POPS: tuple[str, ...] = (
    "CHE", "DNK", "FIN", "FRATNP", "GBRTENW", "GBR_SCO", "ISL", "ITA",
    "NLD", "NOR", "SWE",
)

# ---------------------------------------------------------------------------
# The three pre-registered regimes.
# ---------------------------------------------------------------------------

#: Shift (primary): train <= 2019, test 2020-2024 (COVID break), h = 1..5.
SHIFT = Regime(
    name="shift",
    train_max_year=2019,
    test_years=tuple(range(2020, 2025)),
    horizons=tuple(range(1, 6)),
    pops=SHIFT_POPS,
)

#: Placebo break: train <= 1913, test 1914-1922 (WWI + 1918 flu), h = 1..9.
PLACEBO = Regime(
    name="placebo",
    train_max_year=1913,
    test_years=tuple(range(1914, 1923)),
    horizons=tuple(range(1, 10)),
    pops=PLACEBO_POPS,
)

#: Stable (control): expanding origins T = 1990, 1992, ..., 2014, test
#: T+1..T+5 capped at 2019 — calibration where nothing breaks. One Regime per
#: origin; same populations as SHIFT.
STABLE: tuple[Regime, ...] = tuple(
    Regime(
        name=f"stable_{origin}",
        train_max_year=origin,
        test_years=test,
        horizons=tuple(range(1, len(test) + 1)),
        pops=SHIFT_POPS,
    )
    for origin, test in expanding_origins(1990, 2014, step=2, max_h=5, cap_year=2019)
)

#: Runner configuration: every scoring run iterates this mapping and nothing
#: else, so the pre-registered design is the only entry point to evaluation.
REGIMES: dict[str, Regime | tuple[Regime, ...]] = {
    "shift": SHIFT,
    "stable": STABLE,
    "placebo": PLACEBO,
}
