"""Spark-level tests of IsolationForest / IsolationForestModel (ports of IsolationForestTest + brief coverage)."""
import numpy as np
import pytest
from pyspark.ml.linalg import Vectors

from conftest import auroc, make_df
from pyspark_isolation_forest import IsolationForest, IsolationForestModel


def gaussian_df(spark, n=600, d=4, seed=0, partitions=4):
    X = np.random.default_rng(seed).normal(size=(n, d))
    return make_df(spark, X, partitions=partitions), X


def small_forest(**kw):
    return IsolationForest(numEstimators=kw.pop("numEstimators", 25), **kw)


# ------------------------------------------------------------------ params / API
def test_defaults_match_scala_source():
    f = IsolationForest()
    assert (f.getNumEstimators(), f.getMaxSamples(), f.getMaxFeatures(), f.getContamination(),
            f.getContaminationError(), f.getBootstrap(), f.getRandomSeed()) == (100, 256.0, 1.0, 0.0, 0.0, False, 1)
    assert (f.getFeaturesCol(), f.getPredictionCol(), f.getScoreCol()) == ("features", "predictedLabel", "outlierScore")


def test_fluent_setters_and_getters():
    f = (IsolationForest().setNumEstimators(50).setMaxSamples(128).setMaxFeatures(0.5).setBootstrap(True)
         .setContamination(0.1).setContaminationError(0.001).setRandomSeed(7).setFeaturesCol("f")
         .setPredictionCol("p").setScoreCol("s"))
    assert isinstance(f, IsolationForest)
    assert (f.getNumEstimators(), f.getMaxSamples(), f.getMaxFeatures(), f.getBootstrap(), f.getContamination(),
            f.getContaminationError(), f.getRandomSeed(), f.getFeaturesCol(), f.getPredictionCol(),
            f.getScoreCol()) == (50, 128.0, 0.5, True, 0.1, 0.001, 7, "f", "p", "s")


def test_keyword_constructor_and_setParams():
    f = IsolationForest(numEstimators=10, contamination=0.2)
    assert f.getNumEstimators() == 10 and f.getContamination() == 0.2 and f.getMaxSamples() == 256.0
    f.setParams(maxSamples=64, bootstrap=True)
    assert f.getMaxSamples() == 64.0 and f.getBootstrap() is True


@pytest.mark.parametrize("setter,value", [
    ("setNumEstimators", 0), ("setNumEstimators", -3), ("setMaxSamples", 0.0), ("setMaxSamples", -1),
    ("setContamination", -0.01), ("setContamination", 0.5), ("setContamination", 0.9),
    ("setContaminationError", -0.1), ("setContaminationError", 1.01), ("setMaxFeatures", 0.0),
    ("setMaxFeatures", -1.0), ("setRandomSeed", 0), ("setRandomSeed", -5)])
def test_invalid_params_are_rejected_at_set_time(setter, value):
    with pytest.raises(ValueError):
        getattr(IsolationForest(), setter)(value)


def test_invalid_kwargs_are_rejected_in_constructor():
    with pytest.raises(ValueError):
        IsolationForest(contamination=0.7)


@pytest.mark.parametrize("setter,value", [("setContamination", 0.0), ("setContamination", 0.4999),
                                          ("setContaminationError", 0.0), ("setContaminationError", 1.0),
                                          ("setMaxSamples", 1.0), ("setMaxFeatures", 3), ("setRandomSeed", 1)])
def test_boundary_values_accepted(setter, value):
    getattr(IsolationForest(), setter)(value)


# ------------------------------------------------------------------ fit / transform basics
def test_fit_transform_appends_score_then_prediction_columns(spark):
    df, _ = gaussian_df(spark)
    model = small_forest().fit(df)
    assert isinstance(model, IsolationForestModel)
    out = model.transform(df)
    assert out.columns == ["features", "outlierScore", "predictedLabel"]
    assert out.count() == df.count()
    assert [f.dataType.simpleString() for f in out.schema.fields][1:] == ["double", "double"]


def test_custom_column_names_and_extra_columns_preserved(spark):
    X = np.random.default_rng(1).normal(size=(200, 3))
    df = make_df(spark, X, y=np.arange(200) % 2).withColumnRenamed("features", "vec")
    model = small_forest(featuresCol="vec", scoreCol="s", predictionCol="p", contamination=0.05, maxSamples=100).fit(df)
    out = model.transform(df)
    assert out.columns == ["vec", "label", "s", "p"]
    assert sorted(r["label"] for r in out.collect()) == sorted(float(i % 2) for i in range(200))


def test_scores_lie_in_open_unit_interval_and_orient_outliers_high(spark):
    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal(size=(1000, 3)), rng.normal(loc=15, size=(10, 3))])
    y = np.r_[np.zeros(1000), np.ones(10)]
    pdf = small_forest(numEstimators=50).fit(make_df(spark, X, y)).transform(make_df(spark, X, y)).toPandas()
    assert ((pdf.outlierScore > 0) & (pdf.outlierScore <= 1)).all()
    assert pdf.outlierScore[pdf.label == 1].min() > pdf.outlierScore[pdf.label == 0].mean()
    assert auroc(pdf.outlierScore, pdf.label) > 0.99


def test_model_exposes_resolved_state(spark):
    df, _ = gaussian_df(spark, n=500, d=6)
    m = small_forest(maxSamples=0.5, maxFeatures=0.5).fit(df)
    assert (m.getNumSamples(), m.getNumFeatures(), m.getTotalNumFeatures(), len(m.trees)) == (250, 3, 6, 25)
    assert m.getOutlierScoreThreshold() == -1.0


def test_fit_does_not_mutate_the_estimator(spark):
    df, _ = gaussian_df(spark, n=300)
    est = small_forest(maxSamples=100)
    est.fit(df)
    assert est.getMaxSamples() == 100.0 and est.getNumEstimators() == 25


# ------------------------------------------------------------------ schema / input validation
def test_missing_features_column(spark):
    df, _ = gaussian_df(spark, n=50)
    with pytest.raises(ValueError, match="does not exist"):
        IsolationForest(featuresCol="nope").fit(df)


def test_features_column_must_be_a_vector(spark):
    df = spark.createDataFrame([([1.0, 2.0],)] * 10, ["features"])
    with pytest.raises(ValueError, match="not of required type"):
        IsolationForest().fit(df)


def test_output_columns_must_not_already_exist(spark):
    df, _ = gaussian_df(spark, n=100)
    with pytest.raises(ValueError, match="already exists"):
        IsolationForest(scoreCol="features").fit(df)
    model = small_forest(maxSamples=50).fit(df)
    with pytest.raises(ValueError, match="already exists"):
        model.transform(model.transform(df))


def test_empty_dataset_rejected(spark):
    df, _ = gaussian_df(spark, n=10)
    with pytest.raises(ValueError, match="empty"):
        small_forest().fit(df.filter("1 = 0"))


def test_transform_rejects_wrong_feature_dimension(spark):
    df, _ = gaussian_df(spark, n=200, d=4)
    model = small_forest(maxSamples=100).fit(df)
    wrong = make_df(spark, np.zeros((5, 3)))
    with pytest.raises(Exception, match="did not match the model's training dimension"):
        model.transform(wrong).collect()


def test_sparse_vectors_score_identically_to_dense(spark):
    X = np.random.default_rng(5).normal(size=(300, 4))
    X[X < 0.3] = 0.0
    dense = make_df(spark, X, partitions=2)
    sparse = spark.createDataFrame([(Vectors.sparse(4, {i: v for i, v in enumerate(r) if v != 0.0}),) for r in X],
                                   ["features"]).repartition(2)
    model = small_forest().fit(dense)
    a = np.array(sorted(r["outlierScore"] for r in model.transform(dense).collect()))
    b = np.array(sorted(r["outlierScore"] for r in model.transform(sparse).collect()))
    assert np.allclose(a, b, atol=1e-12)


# ------------------------------------------------------------------ Scala test ports: real datasets
def test_mammography_auroc(mammography):
    """isolationForestMammographyDataTest: AUROC = 0.86 +/- 0.02 (Liu et al. 2008)."""
    contamination = 0.02
    model = IsolationForest(numEstimators=100, bootstrap=False, maxSamples=256, maxFeatures=1.0,
                            contamination=contamination, contaminationError=contamination * 0.01,
                            randomSeed=1).fit(mammography)
    pdf = model.transform(mammography).toPandas()
    assert auroc(pdf.outlierScore, pdf.label) == pytest.approx(0.86, abs=0.02)


def test_mammography_exact_contamination(mammography):
    """isolationForestMammographyExactContaminationDataTest: contaminationError = 0.0 (exact threshold)."""
    contamination = 0.02
    model = IsolationForest(numEstimators=100, maxSamples=256, contamination=contamination,
                            contaminationError=0.0, randomSeed=1).fit(mammography)
    pdf = model.transform(mammography).toPandas()
    assert auroc(pdf.outlierScore, pdf.label) == pytest.approx(0.86, abs=0.02)
    assert pdf.predictedLabel.mean() == pytest.approx(contamination, abs=contamination * 0.01 + 1.0 / len(pdf))


def test_shuttle_scores_and_auroc(shuttle):
    """isolationForestShuttleDataTest: outlier mean 0.61, inlier mean 0.41 (+/-0.02), AUROC > 0.99."""
    contamination = 0.07
    model = IsolationForest(numEstimators=100, maxSamples=256, contamination=contamination,
                            contaminationError=contamination * 0.01, randomSeed=1).fit(shuttle)
    pdf = model.transform(shuttle).toPandas()
    assert pdf.outlierScore[pdf.label == 1].mean() == pytest.approx(0.61, abs=0.02)
    assert pdf.outlierScore[pdf.label == 0].mean() == pytest.approx(0.41, abs=0.02)
    assert auroc(pdf.outlierScore, pdf.label) > 0.99


def test_zero_contamination_gives_all_zero_labels_and_no_threshold(mammography):
    model = IsolationForest(numEstimators=20, contamination=0.0, randomSeed=1).fit(mammography)
    labels = [r["predictedLabel"] for r in model.transform(mammography).select("predictedLabel").collect()]
    assert set(labels) == {0.0} and model.getOutlierScoreThreshold() == -1.0


def test_max_samples_between_one_and_two_is_rejected(spark):
    """Scala: setMaxSamples(1.5) -> floor = 1 sample -> IllegalArgumentException."""
    df, _ = gaussian_df(spark, n=100)
    with pytest.raises(ValueError, match="maxSamples"):
        small_forest(maxSamples=1.5).fit(df)


# ------------------------------------------------------------------ contamination
@pytest.mark.parametrize("n", [500, 4000])
@pytest.mark.parametrize("contamination", [0.01, 0.05, 0.10])
def test_exact_threshold_hits_requested_contamination(spark, n, contamination):
    df, _ = gaussian_df(spark, n=n, d=3, seed=n)
    model = small_forest(numEstimators=20, contamination=contamination, contaminationError=0.0).fit(df)
    pdf = model.transform(df).toPandas()
    assert 0.0 < model.getOutlierScoreThreshold() <= 1.0
    observed = pdf.predictedLabel.mean()
    assert abs(observed - contamination) <= max(contamination * 0.01, 2.0 / n)
    # labels follow the documented rule: score >= threshold
    assert ((pdf.outlierScore >= model.getOutlierScoreThreshold()).astype(float) == pdf.predictedLabel).all()


@pytest.mark.parametrize("contamination", [0.01, 0.05, 0.10])
def test_approximate_threshold_within_requested_error(spark, contamination):
    df, _ = gaussian_df(spark, n=4000, d=3, seed=8)
    err = contamination * 0.1
    model = small_forest(numEstimators=20, contamination=contamination, contaminationError=err).fit(df)
    observed = model.transform(df).toPandas().predictedLabel.mean()
    assert abs(observed - contamination) <= err + 1e-3


# ------------------------------------------------------------------ maxSamples / maxFeatures / bootstrap
@pytest.mark.parametrize("max_samples,expected", [(0.5, 150), (100.0, 100), (1.0, 300), (2.0, 2), (300.0, 300)])
def test_max_samples_resolution(spark, max_samples, expected):
    df, _ = gaussian_df(spark, n=300, d=2)
    m = small_forest(numEstimators=5, maxSamples=max_samples).fit(df)
    assert m.getNumSamples() == expected
    for t in m.trees:
        assert t.num_instances[t.feature < 0].sum() <= expected


def test_max_samples_larger_than_dataset_fails(spark):
    df, _ = gaussian_df(spark, n=100, d=2)
    with pytest.raises(ValueError, match="only 100 samples"):
        small_forest(maxSamples=256).fit(df)


@pytest.mark.parametrize("max_features,expected", [(1.0, 6), (0.5, 3), (0.34, 2), (2.0, 2), (6.0, 6)])
def test_max_features_resolution_and_per_tree_subspace(spark, max_features, expected):
    df, _ = gaussian_df(spark, n=300, d=6)
    m = small_forest(numEstimators=15, maxFeatures=max_features).fit(df)
    assert m.getNumFeatures() == expected
    seen = set()
    for t in m.trees:
        used = set(t.feature[t.feature >= 0].tolist())
        assert len(used) <= expected
        seen |= used
    if expected < 6:
        assert len(seen) > expected      # different trees see different subspaces


@pytest.mark.parametrize("max_features", [0.01, 7.0])
def test_invalid_max_features_fails(spark, max_features):
    df, _ = gaussian_df(spark, n=100, d=6)
    with pytest.raises(ValueError, match="maxFeatures"):
        small_forest(maxFeatures=max_features).fit(df)


def test_bootstrap_trains_and_differs_from_non_bootstrap(spark):
    df, _ = gaussian_df(spark, n=800, d=3)
    a = small_forest(numEstimators=10, maxSamples=100, bootstrap=False).fit(df)
    b = small_forest(numEstimators=10, maxSamples=100, bootstrap=True).fit(df)
    assert [t.to_row() for t in a.trees] != [t.to_row() for t in b.trees]
    assert all(t.num_instances[t.feature < 0].sum() <= 100 for t in b.trees)
    out = b.transform(df).toPandas()
    assert out.outlierScore.between(0, 1).all()


# ------------------------------------------------------------------ randomness
def test_same_seed_same_model_different_seed_different_model(spark):
    df, _ = gaussian_df(spark, n=500)
    rows = lambda m: [t.to_row() for t in m.trees]
    a, b = small_forest(randomSeed=3).fit(df), small_forest(randomSeed=3).fit(df)
    c = small_forest(randomSeed=4).fit(df)
    assert rows(a) == rows(b)
    assert rows(a) != rows(c)
    sa = [r["outlierScore"] for r in a.transform(df).orderBy("features").collect()]
    sb = [r["outlierScore"] for r in b.transform(df).orderBy("features").collect()]
    assert sa == sb


def test_trees_within_a_forest_are_different(spark):
    df, _ = gaussian_df(spark, n=500)
    rows = [str(t.to_row()) for t in small_forest(numEstimators=20).fit(df).trees]
    assert len(set(rows)) == 20


# ------------------------------------------------------------------ small / degenerate data
@pytest.mark.parametrize("n", [2, 3, 10])
def test_very_small_datasets(spark, n):
    X = np.random.default_rng(n).normal(size=(n, 2))
    df = make_df(spark, X, partitions=2)
    m = small_forest(numEstimators=8, maxSamples=float(n) if n > 1 else 1.0, contamination=0.0).fit(df)
    out = m.transform(df).toPandas()
    assert len(out) == n and out.outlierScore.between(0, 1).all() and np.isfinite(out.outlierScore).all()


def test_single_row_dataset_is_rejected(spark):
    df = make_df(spark, np.zeros((1, 2)), partitions=1)
    with pytest.raises(ValueError, match="maxSamples"):
        small_forest(maxSamples=256).fit(df)


def test_constant_feature_never_used_and_scores_valid(spark):
    X = np.random.default_rng(2).normal(size=(400, 4))
    X[:, 2] = 5.0
    df = make_df(spark, X)
    m = small_forest().fit(df)
    for t in m.trees:
        assert 2 not in set(t.feature[t.feature >= 0].tolist())
    assert np.isfinite(m.transform(df).toPandas().outlierScore).all()


def test_all_features_constant_gives_identical_scores(spark):
    df = make_df(spark, np.full((100, 3), 1.5))
    m = small_forest(numEstimators=5, maxSamples=50.0).fit(df)
    scores = {r["outlierScore"] for r in m.transform(df).collect()}
    assert len(scores) == 1 and all(t.num_nodes == 1 for t in m.trees)


def test_duplicate_points(spark):
    X = np.repeat(np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]]), [200, 150, 3], axis=0)
    df = make_df(spark, X)
    m = small_forest(numEstimators=30, maxSamples=128, contamination=0.01).fit(df)
    pdf = m.transform(df.withColumn("k", __import__("pyspark.sql.functions", fromlist=["x"]).monotonically_increasing_id())
                      ).toPandas()
    assert np.isfinite(pdf.outlierScore).all()
    first = pdf.features.map(lambda v: tuple(v.toArray()))
    by_point = pdf.groupby(first).outlierScore.nunique()
    assert (by_point == 1).all()                      # identical points always score identically
    means = pdf.groupby(first).outlierScore.mean()
    assert means[(5.0, 5.0)] > means[(0.0, 0.0)]       # the rare point is the most anomalous


def test_high_dimensional_data(spark):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(500, 60)); X[:5] += 6
    y = np.r_[np.ones(5), np.zeros(495)]
    df = make_df(spark, X, y)
    m = small_forest(numEstimators=40, maxFeatures=0.5).fit(df)
    pdf = m.transform(df).toPandas()
    assert m.getNumFeatures() == 30 and auroc(pdf.outlierScore, pdf.label) > 0.95


def test_partition_count_does_not_break_training(spark):
    X = np.random.default_rng(9).normal(size=(400, 3))
    for p in (1, 8):
        m = small_forest(numEstimators=10).fit(make_df(spark, X, partitions=p))
        assert len(m.trees) == 10


# ------------------------------------------------------------------ distributed-architecture guard
def test_fit_never_collects_the_dataset_to_the_driver(spark, monkeypatch):
    from pyspark.sql import DataFrame
    sizes = []
    original = DataFrame.collect

    def spy(self):
        rows = original(self)
        sizes.append(len(rows))
        return rows

    monkeypatch.setattr(DataFrame, "collect", spy)
    df, _ = gaussian_df(spark, n=3000, d=3)
    small_forest(numEstimators=12, contamination=0.05).fit(df)
    assert sizes and max(sizes) <= 12           # only the trees ever reach the driver
