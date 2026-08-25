"""Size and power checks for the cluster-aware inference layer on synthetic
loss series with a KNOWN cluster structure."""
import numpy as np

from mortcal.inference import dm_wild_cluster, model_confidence_set

G, PER = 20, 30   # 20 populations x 30 units each (2 sexes x 5 horizons x 3 origins)


def _losses(rng, shift=0.0, cluster_sd=0.5):
    groups = np.repeat(np.arange(G), PER)
    common = rng.normal(0, cluster_sd, G)[groups]     # within-population dependence
    a = 1.0 + common + rng.normal(0, 1, G * PER) + shift
    b = 1.0 + common + rng.normal(0, 1, G * PER)
    return a, b, groups


def test_dm_size_under_null_is_controlled():
    rng = np.random.default_rng(1)
    n_rep, rejections = 60, 0
    for _ in range(n_rep):
        a, b, g = _losses(rng)
        rejections += dm_wild_cluster(a, b, g, n_boot=399, rng=rng)["p_value"] < 0.05
    rate = rejections / n_rep
    assert rate <= 0.15, f"null rejection rate {rate:.2f}: wild cluster bootstrap over-rejects"


def test_dm_detects_a_clear_difference():
    rng = np.random.default_rng(2)
    a, b, g = _losses(rng, shift=0.6)
    out = dm_wild_cluster(a, b, g, n_boot=999, rng=rng)
    assert out["p_value"] < 0.05 and out["mean_diff"] > 0 and out["n_clusters"] == G


def test_cluster_se_is_not_the_naive_iid_se():
    """With a strong common within-population component the cluster-robust se
    must be materially LARGER than the iid se -- otherwise clustering is a no-op."""
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(G), PER)
    common = rng.normal(0, 2.0, G)[groups]
    a = common + rng.normal(0, 1, G * PER)
    b = rng.normal(0, 1, G * PER)                      # differential inherits the cluster shock
    d = a - b
    naive_se = d.std(ddof=1) / np.sqrt(d.size)
    out = dm_wild_cluster(a, b, groups, n_boot=99, rng=rng)
    assert out["se"] > 1.5 * naive_se


def test_mcs_keeps_equal_models_and_drops_a_bad_one():
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(G), PER)
    common = rng.normal(0, 0.5, G)[groups]
    base = 1.0 + common
    L = np.column_stack([
        base + rng.normal(0, 1, G * PER),
        base + rng.normal(0, 1, G * PER),
        base + rng.normal(0, 1, G * PER),
        base + rng.normal(0, 1, G * PER) + 1.2,        # clearly worse
    ])
    out = model_confidence_set(L, groups, alpha=0.10, n_boot=500, rng=rng,
                               names=["A", "B", "C", "BAD"])
    assert "BAD" not in out["in_set"]
    assert len(out["in_set"]) >= 2
