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

Cost note (honest accounting, `src/mortcal/uq/conformal.py`): the three
conformal columns do NOT reuse the native fit — every wrapper REFITS the base
family. Split conformal fits twice (once on train-minus-calibration years for
the residuals, once on the full training panel for the interval centre); EnbPI
and copula-conformal fit K=10 trailing-block members plus the full-panel
centre refit (11 fits). The Poisson bootstrap fits 1 + B (B=200 by default).
Per family and origin the classical column therefore costs roughly
1 (native) + 201 (pboot) + 2 (split) + 11 (EnbPI) + 11 (copula) = 226 fits —
trivial for the classical families (all sub-second on a 100×60 panel) but the
same multiplier applies to a neural base, where the K=10 member refits, not
the M=10 ensemble, would dominate. The expensive axis remains (neural families
× ensemble M=10 × conformal members × expanding origins), budgeted for GPU.

Claims discipline (pre-registered): comparisons are stated as contrasts within
admissible sub-grids — e.g. "conformal vs native, classical families only" —
never as full-factorial main effects over a ragged grid.
