"""losses_from_rows must refuse to rank placeholder proper scores.

Conformal cells emit crps/logscore computed from uniform-in-interval samples
and carry scores_secondary=True (addendum 2 §3). On the first real-data
snapshot (2026-08-28) every conformal-family MCS was silently decided on
crps. The guard turns that into an error; the interval score goes through."""
import numpy as np
import pandas as pd
import pytest

from mortcal.inference import losses_from_rows


def _rows():
    rows = []
    rng = np.random.default_rng(0)
    for pop in ("A", "B", "C"):
        for sex in ("f", "m"):
            for model, mech, sec in (("LC", "native", False), ("LC", "split_conf", True),
                                     ("LC", "enbpi", True)):
                rows.append({
                    "regime": "shift", "pop": pop, "sex": sex, "origin": 2019,
                    "model": model, "mechanism": mech, "error": None,
                    "scores_secondary": sec, "n_ages_scored": 100,
                    "crps_h1": rng.uniform(), "crps_h2": rng.uniform(),
                    "winkler95_h1": rng.uniform(), "winkler95_h2": rng.uniform(),
                })
    return pd.DataFrame(rows)


def test_crps_on_a_secondary_arm_raises():
    df = _rows()
    with pytest.raises(ValueError, match="secondary"):
        losses_from_rows(df, loss="crps", arms=[("LC", "native"), ("LC", "split_conf")])
    with pytest.raises(ValueError, match="secondary"):     # even among conformal arms only
        losses_from_rows(df, loss="crps", arms=[("LC", "split_conf"), ("LC", "enbpi")])


def test_interval_score_goes_through_and_native_crps_still_works():
    df = _rows()
    L, groups, names, rep = losses_from_rows(
        df, loss="winkler95", arms=[("LC", "native"), ("LC", "split_conf")])
    assert L.shape == (3 * 2 * 2, 2) and rep["loss"] == "winkler95"
    L2, _, names2, _ = losses_from_rows(df, loss="crps", arms=[("LC", "native")])
    assert names2 == ["LC/native"] and np.isfinite(L2).all()
