"""Model/estimator persistence using Spark ML conventions (no pickle, no JVM model classes)."""
import json
import os

import numpy as np
import pytest
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler

from conftest import make_df
from pyspark_isolation_forest import (ExtendedIsolationForest, ExtendedIsolationForestModel, IsolationForest,
                                      IsolationForestModel)


@pytest.fixture(scope="module")
def data(spark):
    X = np.random.default_rng(0).normal(size=(500, 4))
    X[:5] += 8
    return make_df(spark, X)


def scores(model, df):
    return np.array([r["outlierScore"] for r in model.transform(df).orderBy("features").collect()])


def assert_same_model(a, b, df):
    assert a.uid == b.uid
    for name in ("numEstimators", "maxSamples", "maxFeatures", "contamination", "contaminationError",
                 "bootstrap", "randomSeed", "featuresCol", "predictionCol", "scoreCol"):
        assert a.getOrDefault(name) == b.getOrDefault(name), name
    assert (a.getNumSamples(), a.getNumFeatures(), a.getTotalNumFeatures()) == \
           (b.getNumSamples(), b.getNumFeatures(), b.getTotalNumFeatures())
    assert a.getOutlierScoreThreshold() == b.getOutlierScoreThreshold()
    assert [t.to_row() for t in a.trees] == [t.to_row() for t in b.trees]
    assert np.array_equal(scores(a, df), scores(b, df))


def test_standard_model_roundtrip(spark, data, tmp_path):
    model = IsolationForest(numEstimators=15, maxSamples=100, contamination=0.02, contaminationError=0.001,
                            maxFeatures=0.75, bootstrap=True, randomSeed=9).fit(data)
    path = str(tmp_path / "m")
    model.write().overwrite().save(path)
    loaded = IsolationForestModel.load(path)
    assert isinstance(loaded, IsolationForestModel)
    assert_same_model(model, loaded, data)
    assert loaded.getOutlierScoreThreshold() > 0


def test_extended_model_roundtrip_keeps_resolved_extension_level(spark, data, tmp_path):
    model = ExtendedIsolationForest(numEstimators=15, maxSamples=100, contamination=0.02, maxFeatures=0.75,
                                    randomSeed=4).fit(data)
    path = str(tmp_path / "e")
    model.save(path)
    loaded = ExtendedIsolationForestModel.load(path)
    assert loaded.getExtensionLevel() == model.getExtensionLevel() == 2
    assert_same_model(model, loaded, data)


def test_model_without_threshold_roundtrip(spark, data, tmp_path):
    model = IsolationForest(numEstimators=5, maxSamples=50, contamination=0.0).fit(data)
    model.save(str(tmp_path / "z"))
    loaded = IsolationForestModel.load(str(tmp_path / "z"))
    assert loaded.getOutlierScoreThreshold() == -1.0
    assert {r["predictedLabel"] for r in loaded.transform(data).collect()} == {0.0}


def test_estimator_roundtrip_unset_extension_level_stays_unset(tmp_path):
    est = ExtendedIsolationForest(numEstimators=7, contamination=0.1, bootstrap=True)
    est.save(str(tmp_path / "est"))
    loaded = ExtendedIsolationForest.load(str(tmp_path / "est"))
    assert (loaded.getNumEstimators(), loaded.getContamination(), loaded.getBootstrap()) == (7, 0.1, True)
    assert not loaded.isSet(loaded.extensionLevel)
    est2 = IsolationForest(maxFeatures=0.3, randomSeed=11)
    est2.save(str(tmp_path / "est2"))
    assert IsolationForest.load(str(tmp_path / "est2")).getRandomSeed() == 11


def test_on_disk_format_is_json_metadata_plus_parquet_no_pickle(spark, data, tmp_path):
    path = str(tmp_path / "fmt")
    IsolationForest(numEstimators=4, maxSamples=50, contamination=0.05).fit(data).save(path)
    meta_files = [f for f in os.listdir(f"{path}/metadata") if f.startswith("part-")]
    meta = json.loads(open(f"{path}/metadata/{meta_files[0]}").read())
    assert meta["class"].endswith("isolation_forest_model.IsolationForestModel")
    assert {"outlierScoreThreshold", "numSamples", "numFeatures", "totalNumFeatures", "numTrees", "uid",
            "paramMap"} <= set(meta)
    data_files = os.listdir(f"{path}/data")
    assert any(f.endswith(".parquet") for f in data_files)
    for root, _, files in os.walk(path):
        for f in files:
            assert not f.endswith((".pkl", ".pickle"))


def test_save_twice_requires_overwrite(spark, data, tmp_path):
    model = IsolationForest(numEstimators=4, maxSamples=50).fit(data)
    path = str(tmp_path / "ow")
    model.save(path)
    with pytest.raises(Exception):
        model.save(path)
    model.write().overwrite().save(path)
    IsolationForestModel.load(path)


def test_loading_the_wrong_model_type_fails(spark, data, tmp_path):
    IsolationForest(numEstimators=4, maxSamples=50).fit(data).save(str(tmp_path / "s"))
    with pytest.raises(Exception):
        ExtendedIsolationForestModel.load(str(tmp_path / "s"))


def test_corrupt_tree_count_is_detected(spark, data, tmp_path):
    path = str(tmp_path / "c")
    IsolationForest(numEstimators=4, maxSamples=50).fit(data).save(path)
    meta_file = [f for f in os.listdir(f"{path}/metadata") if f.startswith("part-")][0]
    p = f"{path}/metadata/{meta_file}"
    meta = json.loads(open(p).read()); meta["numTrees"] = 5
    open(p, "w").write(json.dumps(meta))
    crc = f"{path}/metadata/.{meta_file}.crc"          # editing the file by hand invalidates Hadoop's checksum
    if os.path.exists(crc):
        os.remove(crc)
    with pytest.raises(Exception, match="Expected 5 trees"):
        IsolationForestModel.load(path)


# ----------------------------------------------------------------- Pipeline / PipelineModel
def raw_df(spark):
    X = np.random.default_rng(1).normal(size=(400, 3))
    X[:4] += 9
    return spark.createDataFrame([tuple(map(float, r)) for r in X], ["a", "b", "c"])


@pytest.mark.parametrize("cls", [IsolationForest, ExtendedIsolationForest])
def test_pipeline_fit_transform_save_load(spark, tmp_path, cls):
    df = raw_df(spark)
    pipe = Pipeline(stages=[VectorAssembler(inputCols=["a", "b", "c"], outputCol="features"),
                            cls(numEstimators=15, maxSamples=100, contamination=0.02)])
    pm = pipe.fit(df)
    out = pm.transform(df)
    assert out.columns == ["a", "b", "c", "features", "outlierScore", "predictedLabel"]
    path = str(tmp_path / "pm")
    pm.write().overwrite().save(path)
    pm2 = PipelineModel.load(path)
    assert type(pm2.stages[1]).__name__ == cls.__name__ + "Model"
    a = np.array([r["outlierScore"] for r in pm.transform(df).orderBy("a").collect()])
    b = np.array([r["outlierScore"] for r in pm2.transform(df).orderBy("a").collect()])
    assert np.array_equal(a, b)
    assert a[:4].min() > np.median(a)


def test_unfitted_pipeline_save_load(spark, tmp_path):
    pipe = Pipeline(stages=[VectorAssembler(inputCols=["a", "b", "c"], outputCol="features"),
                            IsolationForest(numEstimators=9, contamination=0.1)])
    pipe.write().overwrite().save(str(tmp_path / "p"))
    loaded = Pipeline.load(str(tmp_path / "p"))
    assert loaded.getStages()[1].getNumEstimators() == 9
    assert loaded.getStages()[0].getOutputCol() == "features"
