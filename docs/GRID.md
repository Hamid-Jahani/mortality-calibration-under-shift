# Admissible model-family × UQ-mechanism grid

✓ = primary cell (run and reported). (s) = secondary (run if time; reported in appendix). — = inadmissible (mechanism undefined for that family).

| Family \ Mechanism | native | Poisson bootstrap | deep ensemble | MC dropout | split conformal | EnbPI | copula conformal |
|---|---|---|---|---|---|---|---|
| Lee–Carter (SVD)  | ✓ | ✓ | — | — | ✓ | ✓ | ✓ |
| Poisson-LC        | ✓ | ✓ | — | — | ✓ | ✓ | ✓ |
| CBD (M5)          | ✓ | ✓ | — | — | ✓ | ✓ | ✓ |
| APC / RH (M2-A)   | ✓ | ✓ | — | — | ✓ | ✓ | ✓ |
| sparse VAR        | ✓ | ✓ | — | — | ✓ | ✓ | ✓ |
| multi-output GP   | ✓ | — | (s) | — | ✓ | ✓ | ✓ |
| neural-LC (R–W)   | — | (s) | ✓ | ✓ | ✓ | ✓ | ✓ |
| shallow CNN-LC    | — | (s) | ✓ | ✓ | ✓ | ✓ | ✓ |
| LSTM-on-k_t       | — | (s) | ✓ | ✓ | ✓ | ✓ | ✓ |
| distrib. NB head  | ✓ | — | ✓ | ✓ | ✓ | ✓ | ✓ |

**Primary cells: 50.** Inadmissibility reasons: deterministic fits have no seed
variance (no ensemble/dropout for classical); GP posterior already integrates
parameter uncertainty (bootstrap redundant); neural nets have no model-native
closed-form predictive (their "native" is the distributional head family).

Cost containment: the three conformal columns reuse the SAME underlying fits
as the native/ensemble columns — they are wrappers, not refits. The expensive
axis is (neural families × ensemble M=10 × expanding origins), budgeted for GPU.

Claims discipline (pre-registered): comparisons are stated as contrasts within
admissible sub-grids — e.g. "conformal vs native, classical families only" —
never as full-factorial main effects over a ragged grid.
