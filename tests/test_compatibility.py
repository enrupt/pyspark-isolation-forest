"""Behavioural-fidelity tests against the LinkedIn Scala implementation's documented semantics.

No Scala/JVM implementation is needed: each test either re-implements the Scala rule independently
(scalar, node-by-node, float32 like the Scala code) and compares it with the Spark result, or asserts a
behavioural property read from the Scala source / tests.
"""
import math

import numpy as np
import pytest

from conftest import auroc, make_df
from pyspark_isolation_forest import ExtendedIsolationForest, IsolationForest
from pyspark_isolation_forest.tree import Leaf, Split
from pyspark_isolation_forest.utils import avg_path_length, height_limit


# ------------------------------------------------------------------ independent scalar reference
def scala_path_length(node, x):
    """IsolationTree.pathLength: float32 accumulation, `x < splitValue` goes left, leaf adds c(n)."""
    steps = np.float32(0.0)
    while isinstance(node, Split):
        node = node.left if np.float32(x[node.split_attribute]) < node.split_value else node.right
        steps = np.float32(steps + np.float32(1.0))
    return np.float32(steps + avg_path_length(node.num_instances))


def scala_score(trees, num_samples, x):
    """IsolationForestModel.transform: Array[Float].sum / n  then  Math.pow(2, -pathLength / avgPath)."""
    total = np.float32(0.0)
    for t in trees:
        total = np.float32(total + scala_path_length(t.to_node(), x))
    mean = np.float32(total / np.float32(len(trees)))
    ratio = np.float32(np.float32(-mean) / avg_path_length(num_samples))
    return math.pow(2.0, float(ratio))


@pytest.fixture(scope="module")
def cloud(spark):
    rng = np.random.default_rng(10)
    X = rng.normal(size=(1500, 4))
    return make_df(spark, X), X


@pytest.fixture(scope="module")
def std_model(cloud):
    return IsolationForest(numEstimators=30, maxSamples=128, contamination=0.05, contaminationError=0.0,
                           randomSeed=3).fit(cloud[0])


def test_spark_scores_equal_independent_scala_style_reference(cloud, std_model):
    df, _ = cloud
    rows = std_model.transform(df).limit(60).collect()
    for r in rows:
        ref = scala_score(std_model.trees, std_model.getNumSamples(), r["features"].toArray())
        assert r["outlierScore"] == pytest.approx(ref, rel=0, abs=1e-12)


def test_c_of_n_uses_the_resolved_num_samples_not_the_actual_leaf_sizes(std_model):
    X = np.random.default_rng(0).normal(size=(10, 4)).astype(np.float32)
    s128 = std_model._score_matrix(X)
    std_model._num_samples = 64
    try:
        s64 = std_model._score_matrix(X)
    finally:
        std_model._num_samples = 128
    path = np.zeros(10, dtype=np.float32)
    for t in std_model.trees:
        path = path + t.path_length(X)
    mean = path / np.float32(len(std_model.trees))
    assert np.allclose(s64, 2.0 ** (-(mean / avg_path_length(64)).astype(np.float64)), atol=1e-12)
    assert not np.allclose(s64, s128)


def test_label_rule_is_score_greater_or_equal_threshold(cloud, std_model):
    pdf = std_model.transform(cloud[0]).toPandas()
    thr = std_model.getOutlierScoreThreshold()
    assert ((pdf.outlierScore >= thr).astype(float) == pdf.predictedLabel).all()
    # the threshold itself is attained by a training score, so at least one row sits exactly on it
    assert (pdf.outlierScore == thr).any() and pdf.predictedLabel[pdf.outlierScore == thr].eq(1.0).all()


def test_threshold_is_the_exact_1_minus_contamination_quantile_of_training_scores(cloud, std_model):
    df, _ = cloud
    scores = np.sort(std_model.transform(df).toPandas().outlierScore.to_numpy())
    n = len(scores)
    q = 1 - 0.05
    window = scores[max(int(math.floor(q * n)) - 1, 0): min(int(math.ceil(q * n)) + 1, n)]
    assert std_model.getOutlierScoreThreshold() in window
    ref = std_model.transform(df).stat.approxQuantile("outlierScore", [q], 0.0)[0]
    assert std_model.getOutlierScoreThreshold() == ref


def test_threshold_is_not_a_plain_numpy_percentile_but_close(cloud, std_model):
    scores = std_model.transform(cloud[0]).toPandas().outlierScore.to_numpy()
    assert abs(std_model.getOutlierScoreThreshold() - np.percentile(scores, 95)) < 5e-3


def test_all_trees_respect_the_height_limit_and_leaf_totals(std_model):
    for t in std_model.trees:
        assert t.depth <= height_limit(std_model.getNumSamples())
        assert t.num_instances[t.feature < 0].sum() <= std_model.getNumSamples()


def test_standard_leaves_are_never_empty_but_extended_leaves_can_be(spark):
    X = np.random.default_rng(1).normal(size=(400, 3))
    X[:, 0] = 1.0                                             # a constant feature
    df = make_df(spark, X)
    std = IsolationForest(numEstimators=15, maxSamples=128).fit(df)
    ext = ExtendedIsolationForest(numEstimators=15, maxSamples=128, extensionLevel=0).fit(df)
    assert all((t.num_instances[t.feature < 0] > 0).all() for t in std.trees)
    assert any((t.num_instances[t.left < 0] == 0).any() for t in ext.trees)


def test_extended_level_zero_is_not_equivalent_to_standard_if_constant_features(spark):
    """README: standard IF retries on constant features; EIF (even at level 0) does not."""
    X = np.random.default_rng(2).normal(size=(400, 3))
    X[:, 1] = 4.0
    df = make_df(spark, X)
    std = IsolationForest(numEstimators=20, maxSamples=128).fit(df)
    ext = ExtendedIsolationForest(numEstimators=20, maxSamples=128, extensionLevel=0).fit(df)
    std_uses = any(1 in set(t.feature[t.feature >= 0].tolist()) for t in std.trees)
    ext_uses = any(1 in set(t.indices[t.left >= 0].reshape(-1).tolist()) for t in ext.trees)
    assert not std_uses and ext_uses


# ------------------------------------------------------------------ brief datasets A and B
def test_dataset_a_gaussian_cloud_has_no_clear_outliers(spark):
    X = np.random.default_rng(0).normal(size=(3000, 5))
    df = make_df(spark, X)
    s = IsolationForest(numEstimators=60, maxSamples=256, randomSeed=1).fit(df).transform(df) \
        .toPandas().outlierScore
    assert 0.35 < s.mean() < 0.50                              # inlier mean ~0.41 in Scala's shuttle test
    assert s.max() < 0.80


def test_dataset_b_cloud_plus_far_outliers_is_perfectly_separated(spark):
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(size=(3000, 4)), rng.normal(loc=15, scale=1, size=(30, 4))])
    y = np.r_[np.zeros(3000), np.ones(30)]
    df = make_df(spark, X, y)
    for cls in (IsolationForest, ExtendedIsolationForest):
        pdf = cls(numEstimators=60, maxSamples=256, contamination=0.01, randomSeed=1).fit(df).transform(df).toPandas()
        assert auroc(pdf.outlierScore, pdf.label) > 0.999
        assert pdf.predictedLabel[pdf.label == 1].mean() > 0.9


def test_dataset_i_bootstrap_is_statistically_equivalent_to_no_bootstrap(spark):
    X = np.random.default_rng(3).normal(size=(3000, 3))
    df = make_df(spark, X)
    base = IsolationForest(numEstimators=60, maxSamples=256, randomSeed=1)
    a = base.copy().fit(df).transform(df).toPandas().outlierScore
    b = base.copy().setBootstrap(True).fit(df).transform(df).toPandas().outlierScore
    assert abs(a.mean() - b.mean()) < 0.02
    assert np.corrcoef(a, b)[0, 1] > 0.9


def test_mean_tree_depth_is_close_to_the_height_limit(std_model):
    depths = [t.depth for t in std_model.trees]
    assert np.mean(depths) >= 0.75 * height_limit(128)


def test_model_copy_keeps_threshold_and_trees(cloud, std_model):
    c = std_model.copy({})
    assert c.getOutlierScoreThreshold() == std_model.getOutlierScoreThreshold()
    assert [t.to_row() for t in c.trees] == [t.to_row() for t in std_model.trees]
