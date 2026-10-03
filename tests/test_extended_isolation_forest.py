"""Spark-level tests of ExtendedIsolationForest (ports of ExtendedIsolationForestTest)."""
import numpy as np
import pytest

from conftest import auroc, make_df
from pyspark_isolation_forest import ExtendedIsolationForest, ExtendedIsolationForestModel, IsolationForest


def gaussian_df(spark, n=500, d=4, seed=0):
    X = np.random.default_rng(seed).normal(size=(n, d))
    return make_df(spark, X), X


def eif(**kw):
    return ExtendedIsolationForest(numEstimators=kw.pop("numEstimators", 20), **kw)


def test_defaults_and_extension_level_is_unset_by_default():
    f = ExtendedIsolationForest()
    assert (f.getNumEstimators(), f.getMaxSamples(), f.getRandomSeed()) == (100, 256.0, 1)
    assert not f.isSet(f.extensionLevel)


def test_extension_level_validation():
    with pytest.raises(ValueError):
        ExtendedIsolationForest().setExtensionLevel(-1)
    assert ExtendedIsolationForest().setExtensionLevel(0).getExtensionLevel() == 0


def test_default_extension_level_is_fully_extended_and_stored_on_model(spark):
    df, _ = gaussian_df(spark, d=5)
    m = eif(maxSamples=200).fit(df)
    assert isinstance(m, ExtendedIsolationForestModel)
    assert m.getExtensionLevel() == 4 and all(t.k == 5 for t in m.trees)


@pytest.mark.parametrize("level,k", [(0, 1), (2, 3), (4, 5)])
def test_explicit_extension_level(spark, level, k):
    df, _ = gaussian_df(spark, d=5)
    m = eif(maxSamples=200, extensionLevel=level).fit(df)
    assert m.getExtensionLevel() == level and all(t.k == k for t in m.trees)


def test_extension_level_above_maximum_fails(spark):
    df, _ = gaussian_df(spark, d=3)
    with pytest.raises(ValueError, match=r"must be in \[0, 2\]"):
        eif(maxSamples=100, extensionLevel=3).fit(df)


def test_extension_level_is_relative_to_the_max_features_subspace(spark):
    """README: with 10 features and maxFeatures=0.5 each tree uses 5 features, valid levels are [0, 4]."""
    df, _ = gaussian_df(spark, d=10)
    m = eif(maxSamples=200, maxFeatures=0.5).fit(df)
    assert m.getNumFeatures() == 5 and m.getExtensionLevel() == 4 and all(t.k == 5 for t in m.trees)
    m4 = eif(maxSamples=200, maxFeatures=0.5, extensionLevel=4).fit(df)
    assert m4.getExtensionLevel() == 4
    with pytest.raises(ValueError, match=r"\[0, 4\]"):
        eif(maxSamples=200, maxFeatures=0.5, extensionLevel=5).fit(df)
    for t in m.trees:                       # hyperplanes live in a 5-feature subspace of the 10 original features
        used = set(t.indices[t.left >= 0].reshape(-1).tolist())
        assert len(used) <= 5 and max(used) < 10


def test_default_extension_level_does_not_leak_across_fits(spark):
    """Scala: extendedIsolationForestDefaultExtensionLevelDoesNotLeakAcrossFitsTest."""
    est = eif(maxSamples=100)
    four, _ = gaussian_df(spark, n=300, d=4)
    two, _ = gaussian_df(spark, n=300, d=2)
    m4 = est.fit(four)
    assert not est.isSet(est.extensionLevel)
    m2 = est.fit(two)
    assert (m4.getExtensionLevel(), m2.getExtensionLevel()) == (3, 1)
    assert not est.isSet(est.extensionLevel)


def test_schema_and_score_range(spark):
    df, _ = gaussian_df(spark)
    out = eif(maxSamples=100).fit(df).transform(df)
    assert out.columns == ["features", "outlierScore", "predictedLabel"]
    pdf = out.toPandas()
    assert pdf.outlierScore.between(0, 1).all()


def test_zero_contamination(spark):
    df, _ = gaussian_df(spark)
    m = eif(maxSamples=100, contamination=0.0).fit(df)
    assert m.getOutlierScoreThreshold() == -1.0
    assert set(r["predictedLabel"] for r in m.transform(df).collect()) == {0.0}


def test_exact_contamination(spark):
    df, _ = gaussian_df(spark, n=2000)
    m = eif(maxSamples=128, contamination=0.05, contaminationError=0.0).fit(df)
    observed = m.transform(df).toPandas().predictedLabel.mean()
    assert abs(observed - 0.05) <= 0.05 * 0.01 + 1.0 / 2000


def test_same_seed_same_model(spark):
    df, _ = gaussian_df(spark)
    a, b = eif(maxSamples=100, randomSeed=5).fit(df), eif(maxSamples=100, randomSeed=5).fit(df)
    c = eif(maxSamples=100, randomSeed=6).fit(df)
    rows = lambda m: [t.to_row() for t in m.trees]
    assert rows(a) == rows(b) and rows(a) != rows(c)


def test_bootstrap_and_small_data(spark):
    df, _ = gaussian_df(spark, n=60, d=3)
    m = eif(maxSamples=40, bootstrap=True, extensionLevel=1).fit(df)
    assert m.transform(df).toPandas().outlierScore.between(0, 1).all()
    tiny = make_df(spark, np.random.default_rng(1).normal(size=(2, 2)), partitions=1)
    assert len(eif(maxSamples=2.0, numEstimators=5).fit(tiny).trees) == 5


def test_constant_feature_and_duplicates_are_handled_without_retry(spark):
    X = np.random.default_rng(3).normal(size=(300, 3))
    X[:, 1] = 2.0
    m = eif(maxSamples=100, extensionLevel=1).fit(make_df(spark, X))
    assert np.isfinite(m.transform(make_df(spark, X)).toPandas().outlierScore).all()
    dup = make_df(spark, np.repeat([[0.0, 0.0], [1.0, 1.0]], 100, axis=0))
    assert np.isfinite(eif(maxSamples=100).fit(dup).transform(dup).toPandas().outlierScore).all()


def test_dimension_mismatch_at_transform(spark):
    df, _ = gaussian_df(spark, d=4)
    m = eif(maxSamples=100).fit(df)
    with pytest.raises(Exception, match="did not match"):
        m.transform(make_df(spark, np.zeros((3, 2)))).collect()


# --------------------------------------------------------------- accuracy on real data
def test_mammography_auroc_full_and_zero_extension(mammography):
    """README benchmarks: mammography AUROC ~0.863 (max) / ~0.865 (level 0)."""
    for level in (None, 0):
        kw = {} if level is None else {"extensionLevel": level}
        m = ExtendedIsolationForest(numEstimators=100, maxSamples=256, randomSeed=1, **kw).fit(mammography)
        pdf = m.transform(mammography).toPandas()
        assert auroc(pdf.outlierScore, pdf.label) == pytest.approx(0.865, abs=0.03)


def test_shuttle_auroc(shuttle):
    m = ExtendedIsolationForest(numEstimators=100, maxSamples=256, randomSeed=1, extensionLevel=0).fit(shuttle)
    pdf = m.transform(shuttle).toPandas()
    assert auroc(pdf.outlierScore, pdf.label) > 0.99            # README: 0.9974 at level 0


# --------------------------------------------------------------- standard vs extended (dataset C)
def test_extended_removes_axis_aligned_ghost_regions_on_correlated_data(spark):
    rng = np.random.default_rng(0)
    z = rng.normal(size=(2000, 1))
    X = np.hstack([z, z + 0.1 * rng.normal(size=(2000, 1))])       # dataset C: strongly correlated
    df = make_df(spark, X)
    std = IsolationForest(numEstimators=100, maxSamples=256, randomSeed=1).fit(df)
    ext = ExtendedIsolationForest(numEstimators=100, maxSamples=256, randomSeed=1).fit(df)
    probes = make_df(spark, np.array([[2.0, 2.0], [2.0, -2.0], [-2.0, 2.0], [-2.0, -2.0]]), partitions=1)
    s = {tuple(r["features"].toArray()): r["outlierScore"] for r in std.transform(probes).collect()}
    e = {tuple(r["features"].toArray()): r["outlierScore"] for r in ext.transform(probes).collect()}
    on = [(2.0, 2.0), (-2.0, -2.0)]
    off = [(2.0, -2.0), (-2.0, 2.0)]
    gap_ext = np.mean([e[p] for p in off]) - np.mean([e[p] for p in on])
    gap_std = np.mean([s[p] for p in off]) - np.mean([s[p] for p in on])
    assert gap_ext > 0.05                    # off-diagonal points are clearly more anomalous than on-diagonal ones
    assert gap_ext > gap_std                 # and EIF separates them better than the axis-aligned forest
