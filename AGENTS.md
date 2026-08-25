# AGENTS.md — Mortality / Explainable AI research project

## What this project is

Producing a **publishable actuarial-science paper**, not a product. Working title:

> **Probabilistic Mortality Forecasting under Distribution Shift: Calibration, Uncertainty, and Neural Extensions of Lee–Carter**

Source of the idea: `Datasets and Paper Pathways for Research.docx` (in repo root) — a nine-dataset survey that ranks the Human Mortality Database **first** for a first actuarial publication. This project implements that first idea.

## Locked decisions

Decided with the user on 2026-08-24. Do not silently revisit these.

| Decision | Value |
|---|---|
| Dataset | **Human Mortality Database (HMD)**, `mortality.org`, free after registration |
| Deliverable | **Full research repo + LaTeX paper draft** — not code-only, not protocol-only |
| Target venue | **Actuarial journal** — ASTIN Bulletin / Annals of Actuarial Science / Scandinavian Actuarial Journal. Actuarial framing leads; longevity/annuity impact is the payoff section |
| Contribution | **Calibration under distribution shift ONLY.** Explainability is explicitly deferred to paper #2 despite the folder name |
| Framing | **Reliability audit / benchmark study.** No new architecture. Contribution is the finding + the reusable protocol |
| Stack | **Python + R hybrid.** Python = neural models + evaluation. R (StMoMo/demography) = classical baselines and a numerical oracle. R is **not yet installed** on this machine |
| Compute | Local NVIDIA GPU + CPU, plus a high-RAM CPU server |

### Why "calibration under shift only"

The literature already has neural Lee–Carter **with** intervals (Levantesi et al. 2021; Miyata & Matsuyama 2022). So "we add uncertainty to neural LC" is **not** novel. What nobody has done is **audit whether those intervals hold nominal coverage across a structural break**. The delta is the audit, not the uncertainty. Do not let the paper drift back into a model-proposal.

## The five pre-registered hypotheses

- **H1** — ranking by RMSE ≠ ranking by proper score (CRPS / log score)
- **H2** — nominal 95% intervals under-cover in the shift regime for every family; degradation larger for neural
- **H3** — **joint (path) coverage** over h=1…H is far below marginal coverage, and the literature reports only marginal *(cheapest defensible novelty)*
- **H4** — coverage failure is not uniform in age (concentrates at old ages / infant)
- **H5** — miscalibration propagates to annuity factors and life-expectancy intervals

## Experimental design

**Three regimes:**
1. **Stable** — expanding origin every 2y, h=1…10. Calibration where nothing breaks.
2. **Shift (primary)** — train ≤2019, test 2020–2024 (COVID). **Split fixed before any modelling.**
3. **Placebo break** — train ≤1913, test 1914–1922 (WWI + 1918 flu). Answers "is this just COVID?"

**Crossed design — model family × UQ mechanism are separate factors.** This is what makes it a study rather than a bake-off: it separates "bad coverage is a property of the *architecture*" from "bad coverage is a property of the *uncertainty mechanism bolted on*."

- Models — LC, Poisson-LC (Brouhns), CBD-M5, APC/Renshaw–Haberman, sparse VAR, multi-output GP, neural-LC (Richman–Wüthrich), shallow CNN-LC (Perla et al.), LSTM-on-k_t, distributional Poisson/NB head
- UQ mechanisms — model-native, semiparametric Poisson bootstrap, deep ensemble (M=10), MC dropout, split conformal, EnbPI, copula-conformal (joint)

**Metrics** — RMSE/MAE + e₀ error · Poisson log score · CRPS · Winkler @50/80/95 · marginal coverage · **joint path coverage** · PIT histogram + uniformity test · calibration-by-age · Murphy decomposition (calibration vs resolution) · e_x intervals · annuity ä₆₅ @2% · 99.5% longevity tail quantile · Diebold–Mariano + Model Confidence Set.

## Non-negotiable methodology rules

1. **`PREREGISTRATION.md` is committed and hash-stamped BEFORE the first model fit.** The COVID split is chosen now, not after seeing results.
2. **Expanding-origin splits only. Never a random split.** Mortality panels leak trivially under random splitting.
3. **Hyperparameters tune on an inner *time* split, never on test.**
4. **Every model must emit a predictive distribution, not a point forecast.** Single interface: `fit(panel)` → `predict_dist(horizon, n_samples)` → samples. One evaluation code path for all models.
5. **Validate the evaluation harness on synthetic truth before touching real data.** Simulate from a known LC data-generating process; a correctly-specified model must attain nominal coverage. If the harness cannot confirm that, no result on real data is trustworthy.
6. **Oracle parity** — Python Lee–Carter must match R StMoMo on the same HMD subset to numerical tolerance on α, β, k. This is why the stack is hybrid.
7. Seeds recorded per ensemble member.

## Data notes

HMD requires free registration. Download links live in `literature/` notes and `docs/`. From **Zipped Data Files** take `Deaths_1x1`, `Exposures_1x1`, `Mx_1x1`, and the period life tables. Model Poisson with `log(Exposure)` offset — do not model rates as Gaussian on the raw scale.

Watch: exposure conventions, age boundaries and the 110+ open group, cohort effects, differing observation windows per country, and pre-1950 coverage being patchy (matters for the placebo regime).

## Literature

- `literature/pdf/` — 39 PDFs, the HMD/mortality/UQ track
- `literature/pdf-other-datasets/` — 13 PDFs, the other eight dataset tracks
- `literature/INDEX.md` — categorised, with a one-line "why it matters" per paper
- `literature/MISSING.md` — what could not be downloaded and exactly why

**One missing paper actually matters: Perla, Richman, Scognamiglio & Wüthrich (2021), SAJ.** SSRN serves it free in a browser but 403s scripted fetches. See `MISSING.md`.

## Environment

- Windows 11. Bash tool is Git Bash; PowerShell is 5.1 (no `&&`, no ternary).
- `python` on PATH via bash hits the WindowsApps stub and fails — use `py` or a `uv`-managed interpreter.
- Python 3.14 is installed system-wide; **pin the project to 3.11/3.12 via uv** because torch wheels lag on 3.14.
- `uv` is available at `~/.local/bin/uv.exe`.
- **R is not installed.** Needed for the StMoMo oracle. Install before the baseline-parity work.
- Not a git repository yet.
- arXiv MCP server is broken in this environment (`socksio` missing under a SOCKS proxy). Use `curl` over **https** — plain `http://export.arxiv.org` silently returns nothing.

## Working agreements

- User runs **caveman mode** (terse output, full technical substance) and **explanatory output style** (educational insights inline).
- Do **not** spawn subagents or run Workflow orchestration unless the user explicitly asks.
- Brainstorm → design approval → written spec → plan → implement. Do not jump to code.
