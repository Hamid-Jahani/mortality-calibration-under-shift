"""NaN safety and the runner-row adapter for the inference layer.

Two defects this file exists to prevent, both found 2026-08-26 by running the
real code path:

1. ``model_confidence_set`` INVERTED its answer under the NaN pattern the
   runner actually produces. Measured, 4 models x 20 clusters with 'BAD'
   shifted +1.5: clean -> in_set ['A','B','C'], eliminated [('BAD', 0.000)];
   with ONE hole (model C failing on part of one population) -> in_set
   ['BAD'], eliminated [('A',0.000),('B',0.000),('C',0.000)]. No exception,
   no warning, plausible output, wrong sign. With 9-17% error rows on real
   data this fires on the first analysis.
2. There was no adapter at all from the runner's row format to the
   [n_units, n_models] matrix the inference layer wants, so nothing had ever
   exercised it on runner output.
"""
import numpy as np
import pandas as pd
import pytest

from mortcal.inference import dm_wild_cluster, model_confidence_set, losses_from_rows

G, PER = 8, 4


def _clean(rng, n_models=4, bad_shift=1.5):
    groups = np.repeat(np.arange(G), PER)
    L = (rng.normal(0, 1, size=(G * PER, n_models))
         + np.repeat(rng.normal(0, 0.3, G), PER)[:, None])
    L[:, -1] += bad_shift
    return L, groups


# ---------------------------------------------------------------------------
# 1 — non-finite losses must RAISE, never silently invert
# ---------------------------------------------------------------------------

def test_mcs_raises_on_non_finite_losses():
    rng = np.random.default_rng(0)
    L, groups = _clean(rng)
    names = ["A", "B", "C", "BAD"]
    ok = model_confidence_set(L, groups, names=names, n_boot=100,
                              rng=np.random.default_rng(1))
    assert ok["in_set"] == ["A", "B", "C"]          # sanity: clean case is right
    holed = L.copy()
    holed[5:15, 2] = np.nan                          # model C, part of one pop
    with pytest.raises(ValueError, match="non-finite"):
        model_confidence_set(holed, groups, names=names, n_boot=100,
                             rng=np.random.default_rng(1))


def test_mcs_raises_on_inf_too():
    rng = np.random.default_rng(0)
    L, groups = _clean(rng)
    L[3, 1] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        model_confidence_set(L, groups, n_boot=50, rng=np.random.default_rng(1))


def test_dm_raises_on_non_finite_losses():
    rng = np.random.default_rng(0)
    L, groups = _clean(rng)
    a, b = L[:, 0].copy(), L[:, 3].copy()
    a[7] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        dm_wild_cluster(a, b, groups, n_boot=50, rng=np.random.default_rng(1))


def test_mcs_error_names_the_offending_models():
    rng = np.random.default_rng(0)
    L, groups = _clean(rng)
    L[2, 1] = np.nan
    L[4, 2] = np.nan
    with pytest.raises(ValueError) as e:
        model_confidence_set(L, groups, names=["A", "B", "C", "BAD"], n_boot=50)
    msg = str(e.value)
    assert "B" in msg and "C" in msg, msg


# ---------------------------------------------------------------------------
# 2 — the runner-row adapter
# ---------------------------------------------------------------------------

def _rows(pops=("P1", "P2", "P3"), sexes=("f", "m"), origins=(2019,),
          arms=(("PLC", "native"), ("LC", "native"), ("PLC", "split_conf")),
          H=3, seed=0, holes=()):
    """Runner-shaped rows: one per (regime,pop,sex,origin,model,mechanism)."""
    rng = np.random.default_rng(seed)
    out = []
    for pop in pops:
        for sex in sexes:
            for origin in origins:
                for (m, u) in arms:
                    row = {"regime": "shift", "pop": pop, "sex": sex,
                           "origin": origin, "model": m, "mechanism": u,
                           "h": H, "error": None}
                    broken = (pop, sex, m, u) in holes
                    for k in range(1, H + 1):
                        row[f"crps_h{k}"] = np.nan if broken else float(rng.gamma(2, 1))
                    if broken:
                        row["error"] = "ValueError: synthetic failure"
                    out.append(row)
    return pd.DataFrame(out)


def test_adapter_builds_units_arms_and_groups():
    df = _rows()
    L, groups, names, rep = losses_from_rows(df)
    # units = pop x sex x origin x horizon = 3 * 2 * 1 * 3
    assert L.shape == (18, 3)
    assert len(groups) == 18 and set(groups) == {"P1", "P2", "P3"}
    assert names == ["LC/native", "PLC/native", "PLC/split_conf"]
    assert np.isfinite(L).all()
    assert rep["n_units"] == 18 and rep["n_arms"] == 3


def test_adapter_restricts_to_common_cells_addendum3_s11():
    """§11: contrasts are computed on the INTERSECTION of cells in which every
    compared arm produced a valid row, with the intersection size reported."""
    df = _rows(holes={("P2", "m", "PLC", "split_conf")})
    L, groups, names, rep = losses_from_rows(df)
    assert np.isfinite(L).all(), "no NaN may survive into the loss matrix"
    # P2/m is dropped for EVERY arm, not just the failing one
    assert L.shape == (15, 3)
    assert (np.asarray(groups) == "P2").sum() == 3   # only P2/f survives
    assert rep["n_cells_dropped"] == 1
    assert rep["dropped_cells"] == [("shift", "P2", "m", 2019)]


def test_adapter_reports_the_arms_that_caused_drops():
    df = _rows(holes={("P2", "m", "PLC", "split_conf"),
                      ("P3", "f", "LC", "native")})
    _, _, _, rep = losses_from_rows(df)
    assert rep["n_cells_dropped"] == 2
    assert set(rep["arms_with_failures"]) == {"PLC/split_conf", "LC/native"}


def test_adapter_can_restrict_to_a_subset_of_arms():
    """A contrast within a sub-grid must not be censored by an arm outside it."""
    df = _rows(holes={("P2", "m", "PLC", "split_conf")})
    L, groups, names, rep = losses_from_rows(
        df, arms=[("PLC", "native"), ("LC", "native")])
    assert names == ["LC/native", "PLC/native"]
    assert L.shape == (18, 2), "the failing conformal arm is outside this contrast"
    assert rep["n_cells_dropped"] == 0


def test_adapter_rejects_an_empty_intersection():
    df = _rows(holes={(p, s, "PLC", "split_conf")
                      for p in ("P1", "P2", "P3") for s in ("f", "m")})
    with pytest.raises(ValueError, match="no cells"):
        losses_from_rows(df)


def test_adapter_honours_the_loss_column_choice():
    df = _rows(H=3)
    for k in range(1, 4):
        df[f"logscore_h{k}"] = df[f"crps_h{k}"] * 2.0
    L_c, _, _, _ = losses_from_rows(df, loss="crps")
    L_l, _, _, _ = losses_from_rows(df, loss="logscore")
    np.testing.assert_allclose(L_l, 2.0 * L_c)


def test_adapter_rejects_unknown_loss_column():
    with pytest.raises(ValueError, match="no .* columns"):
        losses_from_rows(_rows(), loss="nonesuch")


def test_adapter_output_feeds_mcs_end_to_end():
    df = _rows(holes={("P2", "m", "PLC", "split_conf")})
    L, groups, names, rep = losses_from_rows(df)
    out = model_confidence_set(L, groups, names=names, n_boot=100,
                               rng=np.random.default_rng(3))
    assert set(out["p_values"]) == set(names)
    assert 1 <= len(out["in_set"]) <= len(names)


def test_adapter_drops_error_rows_even_with_finite_losses():
    """A row carrying an error string is inadmissible whatever its columns say."""
    df = _rows()
    mask = ((df["pop"] == "P1") & (df["sex"] == "f")
            & (df["model"] == "LC") & (df["mechanism"] == "native"))
    df.loc[mask, "error"] = "ValueError: something failed after scoring"
    L, groups, names, rep = losses_from_rows(df)
    assert L.shape == (15, 3)
    assert rep["dropped_cells"] == [("shift", "P1", "f", 2019)]
