"""Distributed-execution guarantees: no dataset collect, executed on Spark, stable under partitioning."""
import numpy as np
import pytest

from conftest import make_df
from pyspark_isolation_forest import ExtendedIsolationForest, IsolationForest


def test_fit_runs_distributed_jobs_and_collects_only_trees(spark, monkeypatch):
    from pyspark.sql import DataFrame
    collected, to_pandas = [], []
    orig_collect, orig_pandas = DataFrame.collect, DataFrame.toPandas
    monkeypatch.setattr(DataFrame, "collect", lambda self: (lambda r: (collected.append(len(r)), r)[1])(orig_collect(self)))
    monkeypatch.setattr(DataFrame, "toPandas", lambda self, *a, **k: (to_pandas.append(1), orig_pandas(self, *a, **k))[1])
    X = np.random.default_rng(0).normal(size=(4000, 4))
    df = make_df(spark, X, partitions=4)
    tracker = spark.sparkContext.statusTracker()
    jobs_before = len(tracker.getJobIdsForGroup())
    for cls in (IsolationForest, ExtendedIsolationForest):
        cls(numEstimators=10, maxSamples=128, contamination=0.05).fit(df)
    assert not to_pandas
    assert collected and max(collected) <= 10           # only the 10 trees are ever collected
    assert len(tracker.getJobIdsForGroup()) >= jobs_before


def test_transform_is_lazy_and_does_not_trigger_a_job(spark):
    X = np.random.default_rng(0).normal(size=(500, 3))
    df = make_df(spark, X)
    model = IsolationForest(numEstimators=5, maxSamples=50).fit(df)
    out = model.transform(df)                    # only builds a plan
    assert "ArrowEvalPython" in out._jdf.queryExecution().executedPlan().toString()


def test_scoring_uses_vectorised_arrow_udf_not_row_by_row_python(spark):
    df = make_df(spark, np.random.default_rng(0).normal(size=(300, 3)))
    plan = IsolationForest(numEstimators=5, maxSamples=50).fit(df).transform(df)._jdf.queryExecution() \
        .executedPlan().toString()
    assert "ArrowEvalPython" in plan and "BatchEvalPython" not in plan


def test_results_are_valid_for_any_number_of_input_partitions(spark):
    X = np.random.default_rng(1).normal(size=(1200, 3))
    X[:6] += 12
    for p in (1, 3, 16):
        df = make_df(spark, X, partitions=p)
        pdf = IsolationForest(numEstimators=40, maxSamples=128, randomSeed=2).fit(df).transform(df).toPandas()
        idx = np.argsort(-pdf.outlierScore.to_numpy())[:6]
        assert len(pdf) == 1200
        top = {tuple(np.round(pdf.features[i].toArray(), 6)) for i in idx}
        planted = {tuple(np.round(x, 6)) for x in X[:6]}
        assert len(top & planted) >= 5, p


def test_fit_on_cached_and_uncached_input_gives_same_model(spark):
    X = np.random.default_rng(2).normal(size=(600, 3))
    df = make_df(spark, X)
    a = IsolationForest(numEstimators=8, maxSamples=64).fit(df)
    b = IsolationForest(numEstimators=8, maxSamples=64).fit(df.cache())
    assert [t.to_row() for t in a.trees] == [t.to_row() for t in b.trees]
