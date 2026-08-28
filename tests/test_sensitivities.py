"""scripts/sensitivities.py on a synthetic runner frame.

What is guarded: the registered population sets partition their panels; error
rows never enter a mean and are counted; horizon-subset slices recompute
coverage from the per-horizon columns and refuse to fabricate a joint path
coverage; conformal arms are reported at 95% only with no proper score in any
leaf; CBD's age support travels with its leaves and never censors the
full-age families' within-mechanism intersection; the addendum 3 §11 cell
accounting adds up; §4 effective cluster counts exclude populations whose
rows all failed; an absent regime is the literal "pending"; the document is
strict JSON (no NaN tokens).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sensitivities as S                                       # noqa: E402
from mortcal.splits import PLACEBO_POPS, SHIFT_POPS             # noqa: E402

ARMS = [("LC", "native"), ("LC", "pboot"), ("LC", "split_conf"), ("LC", "enbpi"),
        ("PLC", "native"), ("PLC", "split_conf"),
        ("CBD", "native"), ("CBD", "split_conf"),
        ("NLC", "ensemble"), ("NLC", "pboot")]        # NLC/pboot is a grid-secondary cell
CONF = {"split_conf", "enbpi", "copula_conf"}
SEXES = ("female", "male")


def make_frame(regime, pops, H, origins=(2019,), errors=None, seed=0):
    """One runner-shaped row per (pop, sex, origin, arm).

    errors: {(pop, model, mechanism): message} — those rows get the error
    string AND poisoned metric values, so a leak into any mean is visible.
    """
    rng = np.random.default_rng(seed)
    errors = errors or {}
    rows = []
    for origin in origins:
        for pop in pops:
            for sex in SEXES:
                for model, mech in ARMS:
                    hcov = rng.uniform(0.6, 1.0, size=H)
                    row = {
                        "regime": regime if len(origins) == 1 else f"{regime}_{origin}",
                        "pop": pop, "sex": sex, "origin": origin, "model": model,
                        "mechanism": mech, "h": H, "error": None,
                        "scores_secondary": mech in CONF,
                        "grid_secondary": (model, mech) in S.SECONDARY,
                        "n_ages_scored": 45 if model == "CBD" else 100,
                        "coverage_95": float(hcov.mean()),
                        "coverage_80": np.nan if mech in CONF else 0.8,
                        "coverage_50": np.nan if mech in CONF else 0.5,
                        "winkler_95": rng.uniform(),
                        "joint_path_coverage_95": float(rng.uniform(0.2, 0.9)),
                        "crps_logmx": rng.uniform(),
                    }
                    row.update({f"coverage95_h{k + 1}": float(hcov[k]) for k in range(H)})
                    row.update({f"winkler95_h{k + 1}": rng.uniform() for k in range(H)})
                    msg = errors.get((pop, model, mech))
                    if msg:
                        row["error"] = msg
                        for c in list(row):
                            if c.startswith("coverage") or c.startswith("joint"):
                                row[c] = 0.0                    # poison
                    rows.append(row)
    return pd.DataFrame(rows)


def arms(slice_):
    return [k for k in slice_ if not k.startswith("_")]


# ---------------------------------------------------------------------------
# registered sets
# ---------------------------------------------------------------------------

def test_registered_sets_partition_their_panels():
    strata = [p for v in S.PLACEBO_STRATA.values() for p in v]
    assert sorted(strata) == sorted(PLACEBO_POPS)
    assert sorted(S.REGISTER_BASED + S.CENSUS_BASED) == sorted(SHIFT_POPS)
    assert not set(S.REGISTER_BASED) & set(S.CENSUS_BASED)
    assert S.REGISTER_BASED == ("CHE", "DNK", "FIN", "ISL", "NOR", "SWE")
    assert S.SHIFT_DROP_2024_H == [1, 2, 3, 4]              # 2024 is horizon 5
    assert S.PLACEBO_DNK_DROP_H == [1, 2, 3, 4, 5, 6, 7]    # 1921-22 are horizons 8-9
    d = S.definitions()
    assert d["shift_population_sets"]["register_based"] == list(S.REGISTER_BASED)
    assert d["shift_population_sets"]["census_based"] == list(S.CENSUS_BASED)
    assert "DATA-PREREQS" in d["shift_population_sets"]["source"]


# ---------------------------------------------------------------------------
# placebo strata and sensitivities
# ---------------------------------------------------------------------------

def test_placebo_strata_and_sensitivities():
    df = make_frame("placebo", PLACEBO_POPS, H=9)
    st = S.placebo_strata(df)
    assert list(st) == ["pooled", "neutral", "belligerent_total", "civilian_only"]
    assert st["pooled"]["LC/native"]["n_cells"] == 22
    assert st["neutral"]["LC/native"]["n_cells"] == 14
    assert st["belligerent_total"]["LC/native"]["n_cells"] == 6
    assert st["civilian_only"]["LC/native"]["n_cells"] == 2
    assert st["civilian_only"]["_meta"]["populations"] == ["GBR_SCO"]
    # strata means recombine to the pooled mean (equal cell weights)
    n = sum(st[k]["LC/native"]["n_cells"] for k in ("neutral", "belligerent_total", "civilian_only"))
    tot = sum(st[k]["LC/native"]["n_cells"] * st[k]["LC/native"]["coverage_95"]
              for k in ("neutral", "belligerent_total", "civilian_only"))
    assert n == 22 and abs(tot / n - st["pooled"]["LC/native"]["coverage_95"]) < 1e-12

    se = S.placebo_sensitivities(df)
    assert list(se) == ["full_panel", "drop_GBR_SCO", "neutral_only",
                        "dnk_full_window", "dnk_drop_1921_1922"]
    assert se["drop_GBR_SCO"]["LC/native"]["n_cells"] == 20
    assert "GBR_SCO" not in se["drop_GBR_SCO"]["_meta"]["populations"]
    assert se["neutral_only"]["LC/native"] == st["neutral"]["LC/native"]
    dnk = df[df["pop"] == "DNK"]
    full, drop = se["dnk_full_window"]["LC/native"], se["dnk_drop_1921_1922"]["LC/native"]
    assert full["n_cells"] == drop["n_cells"] == 2
    g = dnk[(dnk["model"] == "LC") & (dnk["mechanism"] == "native")]
    exp = g[[f"coverage95_h{k}" for k in range(1, 8)]].mean(axis=1).mean()
    assert abs(drop["coverage_95"] - exp) < 1e-12
    assert abs(full["coverage_95"] - g["coverage_95"].mean()) < 1e-12
    assert drop["joint_path_coverage_95"] is None                    # not fabricated
    assert abs(drop["joint_path_coverage_95_lower_bound"] - full["joint_path_coverage_95"]) < 1e-12
    assert drop["horizons"] == [1, 2, 3, 4, 5, 6, 7]
    assert se["dnk_drop_1921_1922"]["_meta"]["dropped_test_years"] == [1921, 1922]


# ---------------------------------------------------------------------------
# shift sensitivities
# ---------------------------------------------------------------------------

def test_shift_sensitivities_slices():
    df = make_frame("shift", SHIFT_POPS, H=5)
    se = S.shift_sensitivities(df)
    assert list(se) == ["full_panel", "drop_2024", "drop_USA_CHL", "register_based", "census_based"]
    fp = se["full_panel"]
    assert fp["LC/native"]["n_cells"] == 40
    # drop_2024: horizons 1-4, joint null, lower bound = full-path joint
    g = df[(df["model"] == "LC") & (df["mechanism"] == "native")]
    exp = g[[f"coverage95_h{k}" for k in (1, 2, 3, 4)]].mean(axis=1).mean()
    d = se["drop_2024"]["LC/native"]
    assert abs(d["coverage_95"] - exp) < 1e-12 and d["horizons"] == [1, 2, 3, 4]
    assert d["joint_path_coverage_95"] is None
    assert abs(d["joint_path_coverage_95_lower_bound"] - fp["LC/native"]["joint_path_coverage_95"]) < 1e-12
    assert se["drop_2024"]["_meta"]["dropped_test_years"] == [2024]
    # with every horizon retained the per-horizon route reproduces coverage_95
    same = S.slice_table(df, horizons=[1, 2, 3, 4, 5])["LC/native"]
    assert abs(same["coverage_95"] - fp["LC/native"]["coverage_95"]) < 1e-12
    # population subsets
    assert se["drop_USA_CHL"]["LC/native"]["n_cells"] == 36
    assert not {"USA", "CHL"} & set(se["drop_USA_CHL"]["_meta"]["populations"])
    assert se["register_based"]["LC/native"]["n_cells"] == 12
    assert se["census_based"]["LC/native"]["n_cells"] == 28
    assert se["register_based"]["_meta"]["populations"] == sorted(S.REGISTER_BASED)
    assert "note" in se["register_based"]["_meta"]


# ---------------------------------------------------------------------------
# scoring discipline
# ---------------------------------------------------------------------------

def test_error_rows_excluded_from_every_mean_and_counted():
    clean = make_frame("shift", SHIFT_POPS, H=5)
    poisoned = make_frame("shift", SHIFT_POPS, H=5,
                          errors={("KOR", "LC", "split_conf"):
                                  "ValueError: panel too short: 8 proper-training years (need >= 10)",
                                  ("TWN", "LC", "native"):
                                  "ValueError: 1000/1000 coefficient draws remain explosive after 100 redraws"})
    ref = S.slice_table(clean[~((clean["pop"] == "KOR") & (clean["mechanism"] == "split_conf"))
                              & ~((clean["pop"] == "TWN") & (clean["model"] == "LC")
                                  & (clean["mechanism"] == "native"))])
    got = S.slice_table(poisoned)
    for arm in ("LC/split_conf", "LC/native"):
        assert got[arm]["coverage_95"] == pytest.approx(ref[arm]["coverage_95"], abs=1e-12)
        assert got[arm]["joint_path_coverage_95"] == pytest.approx(ref[arm]["joint_path_coverage_95"], abs=1e-12)
        assert got[arm]["n_cells"] == 38 and got[arm]["n_error_rows"] == 2
    assert got["PLC/native"]["n_error_rows"] == 0 and got["PLC/native"]["n_cells"] == 40
    assert got["_meta"]["n_error_rows"] == 4
    assert got["_meta"]["error_classes"] == {"machine": 0, "design_floor": 2, "method": 2, "other": 0}
    # horizon-subset route excludes them too
    sub = S.slice_table(poisoned, horizons=[1, 2, 3, 4])
    assert sub["LC/split_conf"]["n_cells"] == 38 and sub["LC/split_conf"]["n_error_rows"] == 2


def test_conformal_arms_reported_at_95_only_and_no_proper_score_anywhere():
    df = make_frame("shift", SHIFT_POPS, H=5)
    doc = S.build({"shift": df, "placebo": None, "stable": None},
                  {"shift": ["results/shift.parquet"]})
    fp = doc["sensitivities"]["shift"]["full_panel"]
    assert fp["LC/split_conf"]["conformal"] is True and fp["LC/native"]["conformal"] is False
    assert fp["LC/split_conf"]["coverage_95"] is not None
    assert fp["NLC/pboot"]["grid_secondary"] is True and fp["NLC/ensemble"]["grid_secondary"] is False
    banned = ("crps", "logscore", "log_score", "pit", "coverage_50", "coverage_80", "winkler")

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k != "_meta" and k != "definitions":
                    assert not any(b in k for b in banned), k
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    for key in ("strata", "sensitivities", "effective_clusters", "common_cell_losses"):
        walk(doc[key])


def test_cbd_age_support_travels_and_never_censors_full_age_families():
    df = make_frame("shift", SHIFT_POPS, H=5,
                    errors={("HRV", "CBD", "split_conf"): "ValueError: panel too short: need >= 10"})
    fp = S.slice_table(df)
    assert fp["CBD/native"]["n_ages_scored"] == 45 and fp["LC/native"]["n_ages_scored"] == 100
    cc = S.common_cell_losses(df)
    wm = cc["within_mechanism:split_conf"]
    assert wm["_meta"]["age_support_blocks"] == {"45": ["CBD"], "100": ["LC", "NLC", "PLC"]}
    assert wm["CBD/split_conf"]["age_support_block"] == 45
    assert wm["LC/split_conf"]["age_support_block"] == 100
    # CBD lost HRV in its own block; LC/PLC (block 100) lost nothing to it
    assert wm["CBD/split_conf"]["n_cells_lost"] == 0 and wm["CBD/split_conf"]["n_cells_full"] == 38
    assert wm["LC/split_conf"]["n_cells_lost"] == 0 and wm["LC/split_conf"]["n_cells"] == 40
    assert wm["_meta"]["n_cells_common"] == {"45": 38, "100": 40}
    # the cell-set intersection over ALL primary arms does drop HRV for everyone
    ap = cc["all_primary_arms"]
    assert ap["LC/native"]["n_cells_lost"] == 2 and ap["LC/native"]["populations_lost"] == ["HRV"]


# ---------------------------------------------------------------------------
# addendum 3 §11 accounting and §4 effective clusters
# ---------------------------------------------------------------------------

def test_common_cell_losses_accounting():
    df = make_frame("shift", SHIFT_POPS, H=5,
                    errors={("KOR", "LC", "split_conf"): "ValueError: panel too short",
                            ("KOR", "LC", "enbpi"): "ValueError: panel too short",
                            ("CHL", "PLC", "native"): "ValueError: lam value too large"})
    cc = S.common_cell_losses(df)
    assert list(cc)[:2] == ["full_panel", "all_primary_arms"]
    assert all(cc["full_panel"][a]["n_cells_lost"] == 0 for a in arms(cc["full_panel"]))
    assert cc["full_panel"]["LC/split_conf"]["n_cells"] == 38
    ap = cc["all_primary_arms"]
    assert ap["_meta"]["n_cells_common"] == 36 and ap["_meta"]["n_cells_panel"] == 40
    assert "NLC/pboot" not in ap["_meta"]["intersecting_arms"]        # secondary never censors
    assert ap["LC/native"]["n_cells_full"] == 40 and ap["LC/native"]["n_cells_lost"] == 4
    assert ap["LC/native"]["populations_lost"] == ["CHL", "KOR"]
    assert ap["LC/split_conf"]["n_cells_full"] == 38 and ap["LC/split_conf"]["n_cells_lost"] == 2
    assert ap["NLC/pboot"]["n_cells"] == 36 and ap["NLC/pboot"]["n_cells_lost"] == 4
    for a in arms(ap):
        assert ap[a]["n_cells_full"] - ap[a]["n_cells_lost"] == ap[a]["n_cells"]
    wf = cc["within_family:LC"]
    assert wf["_meta"]["n_cells_common"] == 38
    assert wf["LC/native"]["n_cells_lost"] == 2 and wf["LC/native"]["populations_lost"] == ["KOR"]
    assert wf["LC/split_conf"]["n_cells_lost"] == 0
    assert set(arms(wf)) == {"LC/native", "LC/pboot", "LC/split_conf", "LC/enbpi"}
    assert cc["within_family:PLC"]["PLC/split_conf"]["n_cells_lost"] == 2       # CHL, via PLC/native
    assert cc["within_family:CBD"]["CBD/native"]["n_cells_lost"] == 0
    wm = cc["within_mechanism:native"]
    assert wm["LC/native"]["n_cells_lost"] == 2 and wm["LC/native"]["populations_lost"] == ["CHL"]
    assert wm["CBD/native"]["n_cells_lost"] == 0                                 # own block


def test_effective_clusters_per_origin():
    pops = SHIFT_POPS[:6]                                   # BEL CHE CHL DNK EST FIN
    errs = {("CHL", m, u): "ValueError: inadmissible: n_train=3 < 15 contiguous training years (addendum 3 §4)"
            for m, u in ARMS}                               # CHL dead at both origins
    errs[("EST", "LC", "native")] = "ValueError: lam value too large"
    df = make_frame("stable", pops, H=5, origins=(1990, 1992), errors=errs)
    ec = S.effective_clusters(df, pops)
    assert list(ec) == ["origin_1990", "origin_1992"]
    m = ec["origin_1990"]["_meta"]
    assert m["origin"] == 1990 and m["n_populations_registered"] == 6
    assert m["n_populations_with_rows"] == 6
    assert m["n_clusters_any_valid_row"] == 5 and "CHL" not in m["populations_any_valid_row"]
    assert m["n_clusters_all_primary_arms"] == 4 and m["populations_all_primary_arms"] == ["BEL", "CHE", "DNK", "FIN"]
    assert m["n_design_floor_rows"] == len(ARMS) * 2 and m["populations_design_floor"] == ["CHL"]
    assert ec["origin_1990"]["LC/native"]["n_clusters"] == 4
    assert ec["origin_1990"]["LC/native"]["populations"] == ["BEL", "CHE", "DNK", "FIN"]
    assert ec["origin_1990"]["PLC/native"]["n_clusters"] == 5
    assert ec["origin_1990"]["PLC/native"]["n_cells"] == 10
    assert ec["origin_1992"]["_meta"]["origin"] == 1992


# ---------------------------------------------------------------------------
# document assembly, pending regimes, strict JSON, CLI
# ---------------------------------------------------------------------------

def test_pending_regimes_and_strict_json():
    df = make_frame("shift", SHIFT_POPS, H=5, errors={("KOR", "LC", "enbpi"): "x"})
    doc = S.build({"shift": df}, {"shift": ["results/shift.parquet"]})
    assert list(doc) == ["contract_version", "generated", "snapshot", "sources", "definitions",
                         "strata", "sensitivities", "effective_clusters", "common_cell_losses"]
    assert doc["snapshot"] is False
    assert doc["strata"]["placebo"] == "pending"
    assert doc["sensitivities"]["placebo"] == "pending"
    assert doc["effective_clusters"]["placebo"] == "pending"
    assert doc["effective_clusters"]["stable"] == "pending"
    assert doc["common_cell_losses"]["placebo"] == "pending"
    assert doc["common_cell_losses"]["stable"] == "pending"
    assert doc["sources"] == {"shift": ["results/shift.parquet"], "placebo": None, "stable": None}
    text = json.dumps(doc, allow_nan=False)                 # raises on NaN/inf
    back = json.loads(text)
    leaf = back["sensitivities"]["shift"]["full_panel"]["LC/enbpi"]
    assert set(S.definitions()["leaf_keys"]) <= set(leaf)
    assert leaf["n_cells"] == 38 and leaf["n_error_rows"] == 2
    # snapshot flag flips on a snapshot basename
    snap = S.build({"shift": df}, {"shift": ["results/_shift_snapshot.parquet"]})
    assert snap["snapshot"] is True


def test_cli_end_to_end(tmp_path):
    df = make_frame("shift", SHIFT_POPS, H=5)
    gp = make_frame("shift", SHIFT_POPS, H=5)
    gp = gp[(gp["model"] == "PLC")].assign(model="GP")       # stand-in GP pass
    df.to_parquet(tmp_path / "shift.parquet", index=False)
    gp.to_parquet(tmp_path / "shift_gp.parquet", index=False)
    out = tmp_path / "sensitivities.json"
    assert S.main(["--results-dir", str(tmp_path), "--out", str(out)]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["snapshot"] is False
    assert len(doc["sources"]["shift"]) == 2                # GP sibling concatenated
    assert "GP/native" in doc["sensitivities"]["shift"]["full_panel"]
    assert doc["strata"]["placebo"] == "pending"
    with pytest.raises(SystemExit):                         # explicit path must exist
        S.main(["--results-dir", str(tmp_path), "--placebo", str(tmp_path / "nope.parquet"),
                "--out", str(out)])
