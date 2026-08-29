# Pre-registration addendum 4 — training-window cap for the multi-output GP

**Registered:** 2026-08-29, during the production sweeps, before any GP cell
on a panel longer than 60 years has produced a result. Applies to the GP
family under every mechanism in every regime. Hash-stamped in its commit like
the earlier addenda.

## What happened

The multi-output GP (exact multitask GP, Kronecker year × age structure,
`mortcal.models.gp`) trains on the trailing block of complete years of each
panel. Its kernel memory scales with (years × ages)²: a 269-year panel is
26,900 points and a 5.8 GB kernel per fit; the EnbPI and copula-conformal
mechanisms refit the family ten times per cell. On the compute node the
second-pass GP sweep stalled after 19 hours with one worker holding 10.7 GB,
the machine swapping, and 14 of 20 populations untouched (six completed:
CHL, HKG, TWN, LUX, and KOR/HRV as design-floor errors). The arm as specified
is infeasible on long panels on the available hardware.

## Rule

The GP family trains on the **trailing 60 complete years** of each training
panel (`max_years = 60`, applied through the runner's per-family kwargs so
every mechanism sees the same family). Panels shorter than 60 complete years
are unaffected. The four populations already completed without the cap have
panels of 28–60 years, so their cells are identical under the rule and are
kept.

## Why 60, and why this is not outcome-driven

- 60 is the longest panel the arm completed without a cap (LUX), so the rule
  changes nothing already computed and is defined by feasibility, not by
  results.
- It is more than twice the 27-year window used by the family's source paper
  (Huynh & Ludkovski 2021), which the pre-registration cites as the reference
  specification for this family.
- Disclosure: before any cap was contemplated, a single diagnostic on one
  population had shown a trailing 40-year window scoring *better* than the
  full 269-year panel (`docs/STATUS.md`, 2026-08-27). The cap chosen here is
  60, not 40; it was not selected by score, and the earlier observation is
  recorded so the reader can judge.
- The GP is one of ten families; its contrasts remain within the registered
  sub-grid discipline. No other family, mechanism, regime, metric, hypothesis
  or inference rule changes.

## Reporting

The results tables state the window for the GP family; the deviations ledger
in the paper's design section carries this addendum with its cause, the
memory figures above, and the disclosure.
