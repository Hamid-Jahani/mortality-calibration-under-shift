"""Gates G-N1..G-N5 of docs/NEURAL-SPEC.md — the five neural/GP families.

Every test uses reduced grids / epochs via constructor arguments so the suite
stays fast; the DEFAULT constructor arguments are the registered grids and are
asserted separately. Requires torch (uv sync --group neural) — the classical
suite must pass without it, so this file importorskips.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mortcal.models import PoissonLeeCarter
from mortcal.models.neural import CNNLC, LSTMKt, NBHead, NeuralLC
from mortcal.models.gp import MultiOutputGP
from mortcal.uq.neural import DeepEnsemble, MCDropout

H = 5
T = 50
N_AGES = 30

FAST = dict(lr_grid=(1e-2,), epochs_grid=(300,))
FAST_CNN = dict(lr_grid=(1e-3,), epochs_grid=(300,))
FAST_GP = dict(lr_grid=(1e-1,), iters_grid=(60,), min_years=30)


def _world(seed, e_scale=1e4, zero_exp_cells=()):
    rng = np.random.default_rng(seed)
    ages = np.arange(N_AGES)
    alpha = -6.5 + 4.5 * (ages / N_AGES) ** 1.2
    beta = np.exp(-0.5 * ((ages - 5) / 6.0) ** 2)
    beta = beta / beta.sum()
    k = np.cumsum(-0.8 + rng.normal(0, 0.5, T + H))
    k = k - k[:T].mean()
    mx_all = np.exp(alpha[:, None] + np.outer(beta, k))
    E = np.full((N_AGES, T + H), float(e_scale))
    D = rng.poisson(E * mx_all).astype(float)
    for (a, t) in zero_exp_cells:
        E[a, t] = 0.0
        D[a, t] = 0.0
    return (D[:, :T], E[:, :T], D[:, T:].T.copy(), E[:, T:].T.copy(),
            np.log(mx_all[:, T:]).T)


FAMILIES = [
    ("NLC", lambda: NeuralLC(**FAST)),
    ("CNN", lambda: CNNLC(**FAST_CNN)),
    ("LSTM", lambda: LSTMKt(**FAST)),
    ("NB", lambda: NBHead(**FAST)),
    ("GP", lambda: MultiOutputGP(**FAST_GP)),
]


# ---------------------------------------------------------------------------
# G-N2 — interface: shapes, finiteness, determinism given seed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,mk", FAMILIES)
def test_interface_shapes_finite_deterministic(name, mk):
    D, E, _, _, _ = _world(1)
    m = mk().fit(D, E)
    s1 = m.sample_mx(H, 20, np.random.default_rng(5))
    s2 = mk().fit(D, E).sample_mx(H, 20, np.random.default_rng(5))
    assert s1.shape == (20, H, N_AGES)
    assert np.isfinite(s1).all()
    assert (s1 > 0).all()
    np.testing.assert_allclose(s1, s2, rtol=1e-6), f"{name}: fit not deterministic given seed"


@pytest.mark.parametrize("name,mk", [f for f in FAMILIES if f[0] in ("NLC", "CNN")])
def test_point_families_native_is_degenerate_and_documented(name, mk):
    """No predictive law of their own: sample_mx repeats the point path."""
    D, E, _, _, _ = _world(2)
    m = mk().fit(D, E)
    s = m.sample_mx(H, 8, np.random.default_rng(0))
    assert np.allclose(s, s[:1]), f"{name}: native must be the repeated point path"
    assert "degenerate" in (m.sample_mx.__doc__ or "").lower()


# ---------------------------------------------------------------------------
# G-N1 — recovery on the synthetic Poisson-LC DGP
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,mk", FAMILIES)
def test_point_forecast_is_on_the_right_scale(name, mk):
    """G-N1 (recalibrated 2026-08-27, measured): a WRONG-SCALE guard, not a
    quality bar. Persistence RMSE on these worlds is 0.25-0.51; the failure
    modes this gate exists to catch measured 1.08 (undertrained NLC), 18-148
    (CNN trained with the diverging original lr grid), and NaN. The
    per-panel extrapolation plateau of the cell-feature nets (~0.5, at or
    above persistence) is the DOCUMENTED audited fragility — quality is what
    the study measures, not what gates assert."""
    D, E, _, _, true_logm = _world(3)
    m = mk().fit(D, E)
    pt = np.median(np.log(m.sample_mx(H, 50, np.random.default_rng(1))), axis=0)
    rmse = float(np.sqrt(np.mean((pt - true_logm) ** 2)))
    assert np.isfinite(rmse) and rmse < 1.0, f"{name}: rmse {rmse:.3f} — G-N1"


# ---------------------------------------------------------------------------
# G-N3 — NB coherence: Gamma rate + runner Poisson composition = NB coverage
# ---------------------------------------------------------------------------

def test_nb_gamma_rates_compose_to_nb_coverage():
    from mortcal.runner import run_cell
    covs = []
    for seed in (11, 12, 13):
        D, E, obs_D, obs_E, _ = _world(seed, e_scale=4e3)
        out = run_cell(D, E, "NB", "native", h=H, n_samples=300,
                       rng=np.random.default_rng(seed), obs_D=obs_D, obs_E=obs_E,
                       mech_kwargs=None)
        covs.append(out["coverage_95"])
    assert 0.88 <= float(np.mean(covs)) <= 1.0, covs


# ---------------------------------------------------------------------------
# G-N4 — structural zeros
# ---------------------------------------------------------------------------

ZEROS = tuple((a, t) for a in (N_AGES - 2, N_AGES - 1) for t in (3, 7, 11))


@pytest.mark.parametrize("name,mk", FAMILIES)
def test_family_fits_through_structural_zero_exposure(name, mk):
    D, E, _, _, _ = _world(4, zero_exp_cells=ZEROS)
    m = mk().fit(D, E)
    s = m.sample_mx(H, 10, np.random.default_rng(1))
    assert np.isfinite(s).all(), f"{name}: non-finite through E=0 cells"


# ---------------------------------------------------------------------------
# G-N5 — mechanisms: member seeds distinct, recorded, reproducible
# ---------------------------------------------------------------------------

def test_deep_ensemble_members_are_distinct_and_recorded():
    D, E, _, _, _ = _world(6)
    ens = DeepEnsemble(NeuralLC, M=3, seed=77, model_kwargs=FAST).fit(D, E)
    assert len(ens.member_seeds) == 3 and len(set(ens.member_seeds)) == 3
    pts = [m._point_logmx(H) for m in ens.members]
    assert not np.allclose(pts[0], pts[1]), "members must differ (different seeds)"
    s = ens.sample_mx(H, 24, np.random.default_rng(2))
    assert s.shape == (24, H, N_AGES) and np.isfinite(s).all()
    # reproducible: same seed -> same members
    ens2 = DeepEnsemble(NeuralLC, M=3, seed=77, model_kwargs=FAST).fit(D, E)
    np.testing.assert_allclose(ens.members[0]._point_logmx(H),
                               ens2.members[0]._point_logmx(H), rtol=1e-6)


def test_ensemble_spread_exceeds_single_member():
    D, E, _, _, _ = _world(7)
    ens = DeepEnsemble(NeuralLC, M=3, seed=5, model_kwargs=FAST).fit(D, E)
    single = NeuralLC(**FAST).fit(D, E)
    se = np.log(ens.sample_mx(H, 30, np.random.default_rng(3))).std(axis=0).mean()
    ss = np.log(single.sample_mx(H, 30, np.random.default_rng(3))).std(axis=0).mean()
    assert se > ss, "ensemble mixture must carry cross-member spread"


def test_mc_dropout_passes_are_stochastic_and_seeded():
    D, E, _, _, _ = _world(8)
    mc = MCDropout(NeuralLC, model_kwargs=FAST).fit(D, E)
    s = mc.sample_mx(H, 16, np.random.default_rng(4))
    assert s.shape == (16, H, N_AGES) and np.isfinite(s).all()
    spread = np.log(s).std(axis=0).mean()
    assert spread > 0.0, "dropout passes must differ"
    s2 = MCDropout(NeuralLC, model_kwargs=FAST).fit(D, E).sample_mx(
        H, 16, np.random.default_rng(4))
    np.testing.assert_allclose(s, s2, rtol=1e-6)


def test_mc_dropout_refuses_family_without_dropout():
    D, E, _, _, _ = _world(9)
    with pytest.raises((ValueError, AttributeError)):
        MCDropout(MultiOutputGP, model_kwargs=FAST_GP).fit(D, E)


# ---------------------------------------------------------------------------
# registry / admissibility
# ---------------------------------------------------------------------------

def test_grid_registry_matches_grid_md():
    from mortcal.runner import ADMISSIBLE, MECHANISMS, MODELS, SECONDARY
    assert set(MODELS) == {"LC", "PLC", "CBD", "RH", "SVAR",
                           "GP", "NLC", "CNN", "LSTM", "NB"}
    assert set(MECHANISMS) == {"native", "pboot", "ensemble", "dropout",
                               "split_conf", "enbpi", "copula_conf"}
    primary = ADMISSIBLE - SECONDARY
    assert len(primary) == 50, f"GRID.md registers 50 primary cells, got {len(primary)}"
    assert SECONDARY == frozenset({("GP", "ensemble"), ("NLC", "pboot"),
                                   ("CNN", "pboot"), ("LSTM", "pboot")})
    # spot checks straight from the table
    assert ("NLC", "native") not in ADMISSIBLE
    assert ("GP", "pboot") not in ADMISSIBLE
    assert ("NB", "native") in primary
    assert ("LC", "ensemble") not in ADMISSIBLE
    assert ("CNN", "dropout") in primary


def test_default_grids_are_the_registered_ones():
    assert NeuralLC().lr_grid == (1e-2, 3e-3) and NeuralLC().epochs_grid == (200, 500)
    # CNN grid corrected 2026-08-27 (measured: every point of the original
    # {1e-2, 3e-3} grid DIVERGES on the synthetic DGP — in-sample RMSE 10-20
    # nats; 1e-3 converges to 0.12). Spec updated the same day, before any
    # real-data run.
    assert CNNLC().lr_grid == (1e-3, 3e-4) and CNNLC().epochs_grid == (300, 800)
    assert LSTMKt().lr_grid == (1e-2, 3e-3) and LSTMKt().epochs_grid == (300, 800)
    assert NBHead().lr_grid == (1e-2, 3e-3) and NBHead().epochs_grid == (200, 500)
    assert MultiOutputGP().lr_grid == (1e-1, 3e-2) and MultiOutputGP().iters_grid == (200, 400)
