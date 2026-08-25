# Pre-registration addendum 1 — data-prerequisite findings

**Registered:** 2026-08-26, still before any model has been fit to real HMD
data. Amends nothing in `PREREGISTRATION.md` (sha256 bbbe3860…); adds
pre-specified sensitivity analyses arising from the HMD documentation check
recorded in `docs/DATA-PREREQS.md`. This addendum is hash-stamped in its
commit like the original.

## A. Placebo regime (train ≤1913, test 1914–1922)

Primary panel unchanged: CHE, DNK, FIN, FRATNP, GBRTENW, GBR_SCO, ISL, ITA,
NLD, NOR, SWE. Belgium's exclusion is confirmed by HMD's own data-series gap
warning for 1914–1918. FRACNP is never used, not even as a sensitivity: HMD
flags its 1919 civilian denominators as implausible.

Pre-specified strata, reported alongside the pooled placebo result:

| Stratum | Populations | Nature of the 1914–22 break |
|---|---|---|
| Neutral / register-based | CHE, DNK, FIN, ISL, NLD, NOR, SWE | pandemic (1918 flu) only — the clean analogue for the 1918-vs-2020 comparison |
| Belligerent, total series | FRATNP, GBRTENW, ITA | pandemic + HMD-reconstructed military deaths at male ages ~18–45 |
| Civilian-only | GBR_SCO | pandemic; military deaths abroad excluded by construction |

Sensitivities (secondary, reported in an appendix):
1. Drop GBR_SCO (ad-hoc population spline HMD itself calls unreliable).
2. Neutral stratum only.
3. DNK: 1921 South-Jutland territorial change is inside the test window;
   report DNK with and without 1921–1922.

## B. Shift regime (train ≤2019, test 2020–2024)

HMD carries no per-year provisional flag; each release re-issues whole
series. The pinned vintage is DOI 10.4054/HMD.Countries.20260615. Known
denominator risks: USA 2020–2025 exposures blend preliminary 2020-census
inputs; CHL exposures rest on the 2017 census pending revision; TWN 2024
deaths provisional per STMF. Register-based populations (Nordics, CHE, NLD,
EST/LVA/LTU) are effectively final.

Sensitivities (secondary), in addition to the already-registered drop-2024:
1. Drop USA and CHL entirely.
2. Register-based vs census-based population split, reported as a contrast.
3. Re-run the shift regime on the next HMD vintage if one is released before
   submission; report any change in headline coverage numbers as a
   vintage-revision effect.

## C. 1918 influenza handling

HMD's Lexis-triangle split for 1918 (Methods Protocol v6, Appendix A) affects
only how raw 1×1 death squares are apportioned into triangles; `Deaths_1x1`
is untouched where raw data were already 1×1. One sentence in the data
appendix; no modelling consequence.

Nothing in this addendum changes hypotheses, primary populations, splits,
horizons, metrics, or inference plan.
