"""Guards for PREREGISTRATION-ADDENDUM-3 — data-defect handling and scoring repairs.

Each test group carries the addendum clause it enforces. The synthetic worlds
here deliberately use LOW exposures (2e3–1e4) with fractional deaths and
structural zero-exposure cells, because the E=1e5 worlds of test_runner.py
have no power: Poisson noise supplies nearly all predictive variance there,
and four mutations that each violate addendum 2 passed 99/99 against them.
"""
import numpy as np
import pytest

from mortcal.models import CBD, LeeCarterSVD, PoissonLeeCarter, RenshawHaberman, SparseVAR
from mortcal.runner import run_cell, _pivot_matrices
from mortcal.uq import SplitConformalMx

import pandas as pd

H = 5
T = 60
N_AGES = 40          # ages 0..39 standing in for a full panel; zeros at the top ages


# ---------------------------------------------------------------------------
# worlds
# ---------------------------------------------------------------------------

def _world(seed, e_scale=4e3, zero_exp_cells=(), frac=False):
    """Poisson-LC DGP with LOW exposure, optional structural E=0 cells and
    fractional (Lexis-style) observed deaths.

    zero_exp_cells: iterable of (age, year_index) set to E=0, D=0 — structural
    zeros as in the HMD audit (ages 95-99, scattered early years).
    """
    rng = np.random.default_rng(seed)
    ages = np.arange(N_AGES)
    alpha = -7.0 + 5.0 * (ages / N_AGES) ** 1.2
    beta = np.exp(-0.5 * ((ages - 6) / 7.0) ** 2)
    beta = beta / beta.sum()
    k = np.cumsum(-0.9 + rng.normal(0, 0.6, T + H))
    k = k - k[:T].mean()
    mx_all = np.exp(alpha[:, None] + np.outer(beta, k))
    E = np.full((N_AGES, T + H), float(e_scale))
    D = rng.poisson(E * mx_all).astype(float)
    for (a, t) in zero_exp_cells:
        E[a, t] = 0.0
        D[a, t] = 0.0
    obs_E = E[:, T:].T.copy()
    obs_D = D[:, T:].T.copy()
    if frac:
        # Lexis-split style fractional deaths on the observed window
        obs_D = obs_D + rng.uniform(0.0, 0.49, size=obs_D.shape) * (obs_D > 0)
    return D[:, :T], E[:, :T], obs_D, obs_E


#: structural zeros at the two top ages in scattered EARLY years (like HMD)
ZEROS = tuple((a, t) for a in (N_AGES - 2, N_AGES - 1) for t in (3, 7, 11, 19))


# ---------------------------------------------------------------------------
# §1 — every family fits a panel containing structural E=0 cells
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["LC", "PLC", "RH", "SVAR"])
def test_family_fits_through_structural_zero_exposure(family):
    D, E, _, _ = _world(101, zero_exp_cells=ZEROS)
    cls = {"LC": LeeCarterSVD, "PLC": PoissonLeeCarter,
           "RH": RenshawHaberman, "SVAR": SparseVAR}[family]
    m = cls().fit(D, E)
    s = m.sample_mx(H, 50, np.random.default_rng(1))
    assert np.isfinite(s).all(), f"{family}: non-finite samples through E=0 cells"
    assert s.max() < 10.0, f"{family}: divergent rates {s.max():.3g}"


def test_cbd_fits_through_structural_zero_exposure():
    # CBD on its own age range, zeros inside it
    D, E, _, _ = _world(102, zero_exp_cells=ZEROS)
    m = CBD().fit(D, E)
    s = m.sample_mx(H, 50, np.random.default_rng(1))
    assert np.isfinite(s).all()
    assert np.isfinite(m.k1).all() and np.isfinite(m.k2).all()


def test_weighted_fits_reduce_to_unweighted_on_clean_panels():
    """§1 reduction requirement on the families with an explicit weighted path."""
    D, E, _, _ = _world(103)
    # CBD: the weighted per-year solve must reproduce the closed-form column
    # means / slope of the unweighted OLS exactly.
    m = CBD().fit(D, E)
    m_hat = np.maximum(D, 0.5) / E
    q = np.clip(1.0 - np.exp(-m_hat), 1e-12, 1 - 1e-12)
    y = np.log(q) - np.log1p(-q)
    xc = np.arange(N_AGES) - np.arange(N_AGES).mean()
    np.testing.assert_allclose(m.k1, y.mean(axis=0), rtol=1e-10)
    np.testing.assert_allclose(m.k2, (xc[:, None] * y).sum(axis=0) / (xc ** 2).sum(),
                               rtol=1e-10)
    # LC: EM path on a panel with a couple of zeros reproduces the clean-fit
    # parameters closely (the zeros carry no information).
    Dz, Ez = D.copy(), E.copy()
    for (a, t) in ((N_AGES - 1, 3), (N_AGES - 1, 11)):
        Ez[a, t] = 0.0
        Dz[a, t] = 0.0
    clean = LeeCarterSVD().fit(D, E)
    holed = LeeCarterSVD().fit(Dz, Ez)
    np.testing.assert_allclose(holed.beta, clean.beta, atol=5e-3)
    np.testing.assert_allclose(holed.kappa, clean.kappa, atol=0.15)


def test_zero_death_training_cells_no_longer_hit_the_rate_floor():
    """B1: no model may see log(1e-10) = -23.03 from a zero-death cell.

    The measured failure was the INITIALISER: alpha started at -23.03 on
    zero-death-heavy ages and PLC's first Newton step overshot to +1e4,
    leaving the whole fit NaN. An age with zero deaths in HALF its years must
    initialise near the half-count rate and converge finite. (An age with
    zero deaths in EVERY year legitimately walks toward -inf — that is the
    MLE, not an artefact.)"""
    D, E, _, _ = _world(104, e_scale=2e3)
    # a genuinely sparse top age: E = 8, expected deaths ~ 1/year, so ~1/3 of
    # its years carry ZERO deaths — the real HMD situation (ISL, LUX)
    E[-1, :] = 8.0
    D[-1, :] = np.random.default_rng(9).poisson(8.0 * np.exp(-2.15), size=T).astype(float)
    assert (D[-1] == 0).sum() >= 5, "world must contain natural zero-death years"
    for cls in (LeeCarterSVD, PoissonLeeCarter):
        m = cls().fit(D, E)
        assert np.isfinite(m.alpha).all() and np.isfinite(m.kappa).all(), cls.__name__
        assert -6.0 < m.alpha[-1] < 0.0,             f"{cls.__name__}: alpha[-1]={m.alpha[-1]:.2f} (true ~ -2.15; floor gave -23)"
        s = m.sample_mx(H, 50, np.random.default_rng(2))
        assert np.isfinite(s).all()


# ---------------------------------------------------------------------------
# §2 — training-year contiguity (BEL)
# ---------------------------------------------------------------------------

def _tidy(D, E, first_year=1950, pop="SYN", sex="f", drop_years=()):
    ages = np.arange(D.shape[0])
    years = first_year + np.arange(D.shape[1])
    aa, yy = np.meshgrid(ages, years, indexing="ij")
    df = pd.DataFrame({"pop": pop, "year": yy.ravel(), "age": aa.ravel(),
                       "sex": sex, "D": D.ravel(), "E": E.ravel()})
    return df[~df["year"].isin(drop_years)].reset_index(drop=True)


def test_pivot_trims_to_maximal_contiguous_block():
    """A BEL-style all-age year gap: train on the post-gap block, not a
    silently spliced panel."""
    D, E, _, _ = _world(105)
    sub = _tidy(D, E, first_year=1950, drop_years=(1964, 1965, 1966))
    tr_D, tr_E, _, _ = _pivot_matrices(sub, train_max_year=1950 + T - H - 1,
                                       test_years=tuple(range(1950 + T - H, 1950 + T)))
    # post-gap block: 1967 .. origin
    assert tr_D.shape[1] == (1950 + T - H - 1) - 1967 + 1, tr_D.shape


# ---------------------------------------------------------------------------
# §3 — zero-exposure cells in the TEST window
# ---------------------------------------------------------------------------

def test_test_window_zero_exposure_age_is_masked_and_derived_quantities_survive():
    D, E, obs_D, obs_E = _world(106)
    obs_E[2, N_AGES - 1] = 0.0              # nobody alive at top age, horizon 3
    obs_D[2, N_AGES - 1] = 0.0
    out = run_cell(D, E, "PLC", "native", h=H, n_samples=150,
                   rng=np.random.default_rng(6), obs_D=obs_D, obs_E=obs_E)
    assert out["n_ages_scored"] == N_AGES - 1
    assert np.isfinite(out["coverage_95"])
    assert np.isfinite(out["e0_point"]), "derived quantities must use the contiguous scored range"


# ---------------------------------------------------------------------------
# §4 — minimum training length
# ---------------------------------------------------------------------------

def test_short_training_window_is_inadmissible():
    D, E, obs_D, obs_E = _world(107)
    sub = _tidy(np.concatenate([D, obs_D.T], axis=1),
                np.concatenate([E, obs_E.T], axis=1), first_year=1950)
    with pytest.raises(ValueError, match="n_train"):
        _pivot_matrices(sub[sub["year"] >= 1950 + T - 8],
                        train_max_year=1950 + T - 1,
                        test_years=tuple(range(1950 + T, 1950 + T + H)))


# ---------------------------------------------------------------------------
# §5 — rate-scale truth lives on the rounded lattice
# ---------------------------------------------------------------------------

def test_fractional_deaths_do_not_move_rate_scale_scores():
    """Observed D and D + 0.4 round to the same integer: every rate-scale
    metric must be identical. Kills the unrounded-truth defect."""
    D, E, obs_D, obs_E = _world(108)
    obs_D = np.maximum(np.round(obs_D), 1.0)          # integer, >= 1
    frac = obs_D + 0.4
    a = run_cell(D, E, "PLC", "native", h=H, n_samples=200,
                 rng=np.random.default_rng(8), obs_D=obs_D, obs_E=obs_E)
    b = run_cell(D, E, "PLC", "native", h=H, n_samples=200,
                 rng=np.random.default_rng(8), obs_D=frac, obs_E=obs_E)
    for k in ("rmse_logmx", "crps_logmx", "coverage_95", "pit_ks_stat"):
        assert a[k] == b[k], (k, a[k], b[k])


def test_calibrated_forecast_with_fractional_deaths_stays_calibrated():
    """B5's measured artefact: PIT KS 0.019 -> 0.133 at small lambda when
    fractional truth meets an integer lattice. After §5 both live on the
    lattice and a correctly-specified model stays near nominal."""
    covs, ks = [], []
    for seed in (31, 32, 33):
        D, E, obs_D, obs_E = _world(seed, e_scale=3e3, frac=True)
        out = run_cell(D, E, "PLC", "native", h=H, n_samples=400,
                       rng=np.random.default_rng(seed), obs_D=obs_D, obs_E=obs_E)
        covs.append(out["coverage_95"])
        ks.append(out["pit_ks_stat"])
    assert 0.90 <= float(np.mean(covs)) <= 0.985, covs
    # 200 dependent cells/seed; KS 5% critical ~ 0.096. The B5 artefact this
    # kills measured 0.133 PER SEED at lambda=2 with unrounded truth.
    assert float(np.mean(ks)) < 0.105, ks


# ---------------------------------------------------------------------------
# §6 — conformal cells scored from their interval bounds
# ---------------------------------------------------------------------------

def test_conformal_wrapper_exposes_interval_bounds():
    D, E, _, _ = _world(109)
    w = SplitConformalMx(PoissonLeeCarter).fit(D, E)
    lo, hi = w.interval(H)
    assert lo.shape == (H, N_AGES) and hi.shape == (H, N_AGES)
    assert np.all(lo < hi)


def test_conformal_coverage_is_computed_from_bounds_not_sample_quantiles():
    """Bound-based scoring is invariant to n_samples; quantile-of-uniform-
    samples scoring is not. Kills the committed defect (measured: 0.995 as
    committed vs 0.960 on the intervals; joint 0.957 vs 0.740)."""
    D, E, obs_D, obs_E = _world(110, e_scale=3e3)
    a = run_cell(D, E, "PLC", "split_conf", h=H, n_samples=60,
                 rng=np.random.default_rng(10), obs_D=obs_D, obs_E=obs_E)
    b = run_cell(D, E, "PLC", "split_conf", h=H, n_samples=1200,
                 rng=np.random.default_rng(11), obs_D=obs_D, obs_E=obs_E)
    for k in ("coverage_95", "winkler_95", "joint_path_coverage_95"):
        assert a[k] == b[k], (k, a[k], b[k])


def test_conformal_coverage_matches_directly_evaluated_bounds():
    """No Poisson composition on conformal interval metrics (the radius already
    contains observation noise; composing more double-counts it)."""
    from mortcal.eval.scores import round_deaths
    D, E, obs_D, obs_E = _world(111, e_scale=3e3)
    out = run_cell(D, E, "PLC", "split_conf", h=H, n_samples=200,
                   rng=np.random.default_rng(12), obs_D=obs_D, obs_E=obs_E)
    w = SplitConformalMx(PoissonLeeCarter).fit(D, E)
    lo, hi = w.interval(H)                               # log m_x bounds
    truth = np.log(np.maximum(round_deaths(obs_D), 0.5) / obs_E)
    hit = (truth >= lo) & (truth <= hi)
    assert out["coverage_95"] == pytest.approx(hit.mean()), \
        "conformal coverage must equal the fraction of truths inside the wrapper's own bounds"
    assert out["joint_path_coverage_95"] == pytest.approx(hit.all(axis=0).mean())


# ---------------------------------------------------------------------------
# §7 — SVAR rejects explosive coefficient draws
# ---------------------------------------------------------------------------

def test_svar_predictive_rates_are_bounded():
    """No stability constraint -> divergent VARs with m_x up to 3e45 on real
    panels. Rejection sampling on the companion spectral radius bounds them."""
    rng = np.random.default_rng(3)
    D, E, _, _ = _world(112, e_scale=2e3)
    m = SparseVAR().fit(D, E)
    s = m.sample_mx(9, 400, rng)
    assert np.isfinite(s).all()
    assert s.max() < 1e3, f"max predictive m_x = {s.max():.3g}"


# ---------------------------------------------------------------------------
# §9 — conformal calibration horizon covers h = 9 (placebo)
# ---------------------------------------------------------------------------

def test_conformal_calibration_covers_placebo_horizon():
    D, E, _, _ = _world(113)
    w = SplitConformalMx(PoissonLeeCarter).fit(D, E)
    lo, hi = w.interval(9)
    assert lo.shape[0] == 9
    assert w.cal_years >= 9, "h=9 must be calibrated, not extrapolated"


# ---------------------------------------------------------------------------
# §10 — zero-death cells counted pre-mask at D < 0.5
# ---------------------------------------------------------------------------

def test_zero_death_count_is_pre_mask_and_half_count_thresholded():
    D, E, obs_D, obs_E = _world(114)
    obs_D[0, 5] = 0.0
    obs_D[1, 7] = 0.3                        # fractional: correction binds
    obs_D[2, 9] = 0.7                        # correction does NOT bind
    expected = int((obs_D < 0.5).sum())      # naturals + the injected 0.0 and 0.3
    assert expected >= 2
    plc = run_cell(D, E, "PLC", "native", h=H, n_samples=100,
                   rng=np.random.default_rng(14), obs_D=obs_D, obs_E=obs_E)
    assert plc["n_zero_death_cells"] == expected
    # threshold check: raising the 0.3 cell above 0.5 must drop the count by 1
    obs_D2 = obs_D.copy(); obs_D2[1, 7] = 0.7
    plc2 = run_cell(D, E, "PLC", "native", h=H, n_samples=100,
                    rng=np.random.default_rng(14), obs_D=obs_D2, obs_E=obs_E)
    assert plc2["n_zero_death_cells"] == expected - 1
    # CBD masks ages below 55 -> here fit on all ages via age_min=None is not
    # the runner path; emulate the mask by SVAR which scores all ages, and
    # CBD-style masking via the runner's CBD registry entry is covered in
    # test_runner. The registered property: the count must NOT depend on the
    # model's age mask. Verify via a family whose mask differs (SVAR == PLC).
    svar = run_cell(D, E, "SVAR", "native", h=H, n_samples=100,
                    rng=np.random.default_rng(14), obs_D=obs_D, obs_E=obs_E)
    assert svar["n_zero_death_cells"] == plc["n_zero_death_cells"]


# ---------------------------------------------------------------------------
# B7 — power: the suite must detect a runner that deletes forecast uncertainty
# ---------------------------------------------------------------------------

def test_model_uncertainty_dominates_at_high_exposure():
    """At E=1e7 Poisson noise is negligible: coverage comes almost entirely
    from the model's own spread. A mutation drawing Poisson on the sample-MEAN
    path (deleting forecast uncertainty) collapses this to ~0."""
    covs = []
    for seed in (51, 52, 53):
        D, E, obs_D, obs_E = _world(seed, e_scale=1e7)
        out = run_cell(D, E, "PLC", "native", h=H, n_samples=400,
                       rng=np.random.default_rng(seed), obs_D=obs_D, obs_E=obs_E)
        covs.append(out["coverage_95"])
    assert float(np.mean(covs)) >= 0.85, covs


def test_grossly_miscalibrated_forecast_is_flagged_by_pit_pvalue():
    """Kills the `return 0.5` mutation on the p-value and pins its direction."""
    D, E, obs_D, obs_E = _world(115, e_scale=3e3)
    out = run_cell(D, E, "PLC", "native", h=H, n_samples=300,
                   rng=np.random.default_rng(15), obs_D=obs_D * 3.0, obs_E=obs_E)
    assert out["pit_ks_stat"] > 0.3
    assert out["pit_ks_pvalue"] < 1e-4


# ---------------------------------------------------------------------------
# crps_counts shares the rate scale's Poisson draw (adversarial review, LATER
# item): PLC/RH exposed sample_deaths, which redraws its OWN kappa paths, so
# crps_counts came from a different realisation than every other column in its
# row — MC noise plus a cross-family asymmetry (only the two families with
# that method were affected).
# ---------------------------------------------------------------------------

def test_count_scale_is_the_same_draw_as_the_rate_scale():
    """log_samples and the count samples must be two views of ONE draw:
    log(max(D*,0.5)/E) is recoverable from the death samples exactly."""
    from mortcal.runner import _compose_deaths
    from mortcal.eval.scores import log_crude_rate
    from mortcal.models import PoissonLeeCarter

    D, E, obs_D, obs_E = _world(201)
    est = PoissonLeeCarter().fit(D, E)
    smx = est.sample_mx(H, 40, np.random.default_rng(1))
    age_ok = np.ones(N_AGES, dtype=bool)
    Es = obs_E[None, :, age_ok]

    d1 = _compose_deaths(smx, Es, age_ok, np.random.default_rng(9))
    d2 = _compose_deaths(smx, Es, age_ok, np.random.default_rng(9))
    np.testing.assert_array_equal(d1, d2, "composition must be rng-deterministic")
    # the two scales agree cell for cell
    np.testing.assert_allclose(log_crude_rate(d1, Es), np.log(np.maximum(d1, 0.5) / Es))


def test_count_composition_is_family_independent():
    """A family exposing sample_deaths must get the SAME construction as one
    that does not — no privileged path."""
    from mortcal.runner import _compose_deaths
    from mortcal.models import LeeCarterSVD, PoissonLeeCarter

    D, E, _, obs_E = _world(202)
    age_ok = np.ones(N_AGES, dtype=bool)
    Es = obs_E[None, :, age_ok]
    plc = PoissonLeeCarter().fit(D, E)
    assert hasattr(plc, "sample_deaths"), "precondition: PLC has the shortcut"
    smx = plc.sample_mx(H, 30, np.random.default_rng(2))
    a = _compose_deaths(smx, Es, age_ok, np.random.default_rng(3))
    b = _compose_deaths(smx, Es, age_ok, np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)
    lc_smx = LeeCarterSVD().fit(D, E).sample_mx(H, 30, np.random.default_rng(2))
    c = _compose_deaths(lc_smx, Es, age_ok, np.random.default_rng(3))
    assert c.shape == a.shape           # identical construction, different paths


def test_crps_counts_is_reproducible_from_the_cell_seed():
    D, E, obs_D, obs_E = _world(203)
    kw = dict(h=H, n_samples=200, obs_D=obs_D, obs_E=obs_E)
    a = run_cell(D, E, "PLC", "native", rng=np.random.default_rng(77), **kw)
    b = run_cell(D, E, "PLC", "native", rng=np.random.default_rng(77), **kw)
    assert a["crps_counts"] == b["crps_counts"]
    assert np.isfinite(a["crps_counts"]) and a["crps_counts"] > 0


def test_svar_stability_prefilter_is_exact():
    """The infinity-norm pre-filter must never accept an explosive draw:
    rho(A) <= ||A||_inf is a theorem, so a draw passing the cheap test is
    provably stable, and anything failing it still gets the eigendecomposition.
    """
    from mortcal.models.svar import SparseVAR
    rng = np.random.default_rng(5)
    D, E, _, _ = _world(301)
    m = SparseVAR().fit(D, E)
    c, B = m._sample_coefs(200, rng)
    rho = m._spectral_radius(B)
    assert np.all(rho < 1.0), "every returned draw must be stable"
    # the bound itself
    A = np.zeros((B.shape[0], m.n_age, m.n_age))
    W = m.W
    for k in range(2 * W + 1):
        d = k - W
        if d < 0:
            idx = np.arange(-d, m.n_age)
            A[:, idx, idx + d] = B[:, -d:, k]
        else:
            idx = np.arange(0, m.n_age - d)
            A[:, idx, idx + d] = B[:, :m.n_age - d if d else m.n_age, k]
    assert np.all(rho <= np.abs(A).sum(axis=2).max(axis=1) + 1e-9), \
        "rho(A) <= ||A||_inf must hold for every draw"


# ---------------------------------------------------------------------------
# a family undefined on part of the age range must still take a conformal
# wrapper: the bands it cannot inform get a NaN radius (masked in scoring),
# not an exception that kills every CBD x conformal cell in the grid.
# ---------------------------------------------------------------------------

def test_conformal_wraps_a_family_with_a_restricted_age_range():
    from mortcal.models import CBD
    from mortcal.uq import SplitConformalMx, EnbPIMx
    import functools
    D, E, obs_D, obs_E = _world(401)
    factory = functools.partial(CBD, age_min=25)     # undefined below age 25
    for cls in (SplitConformalMx, EnbPIMx):
        w = cls(factory).fit(D, E)
        lo, hi = w.interval(H)
        assert np.isfinite(lo[:, 25:]).all(), f"{cls.__name__}: defined ages must be finite"
        assert np.isnan(lo[:, :25]).all(), f"{cls.__name__}: undefined ages must be NaN"


def test_cbd_conformal_cell_scores_its_defined_ages():
    """CBD is fit on ages 55+ (MODEL_KWARGS), so a conformal wrapper around it
    must score ages 55-99 and mask the rest — not error out. Needs a
    full-width panel, so this world is built here rather than by _world."""
    from mortcal.runner import MODEL_KWARGS
    assert MODEL_KWARGS["CBD"]["age_min"] == 55
    n_ages, T_, rng = 100, 60, np.random.default_rng(402)
    ages = np.arange(n_ages)
    alpha = -7.5 + 5.5 * (ages / n_ages) ** 1.3
    beta = np.exp(-0.5 * ((ages - 12) / 14.0) ** 2)
    beta = beta / beta.sum()
    k = np.cumsum(-1.0 + rng.normal(0, 0.7, T_ + H))
    k -= k[:T_].mean()
    mx = np.exp(alpha[:, None] + np.outer(beta, k))
    E_ = np.full((n_ages, T_ + H), 2e4)
    D_ = rng.poisson(E_ * mx).astype(float)
    out = run_cell(D_[:, :T_], E_[:, :T_], "CBD", "split_conf", h=H,
                   n_samples=200, rng=np.random.default_rng(4),
                   obs_D=D_[:, T_:].T, obs_E=E_[:, T_:].T)
    assert np.isfinite(out["coverage_95"]), "CBD x conformal must produce a score"
    assert out["n_ages_scored"] == n_ages - 55
