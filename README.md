# pyspark-isolation-forest

A **native PySpark** implementation of [`linkedin/isolation-forest`](https://github.com/linkedin/isolation-forest):
the standard **Isolation Forest** and the **Extended Isolation Forest** (random hyperplane splits), as
`Estimator`/`Model` pairs that work in `Pipeline`s. There is no Py4J delegation to the Scala library and no
scikit-learn dependency: the algorithms are implemented in Python/NumPy and distributed with Spark's
Arrow-based DataFrame APIs.

The port follows the Scala *source* (not the academic paper) wherever they differ - see
[`PORTING_ANALYSIS.md`](PORTING_ANALYSIS.md) and [`COMPATIBILITY.md`](COMPATIBILITY.md).

## Installation and compatibility

```bash
pip install .            # or: pip install -e ".[test]"
```

| Component | Tested with | Notes |
|---|---|---|
| Spark / PySpark | **3.5.5** (classic, non-Connect) | uses `mapInArrow`, `applyInPandas`, `pandas_udf` |
| Python | 3.12 | `requires-python >= 3.9` |
| JVM | **JDK 17** | **JDK 21 does not work with Spark 3.5** for any Arrow UDF (Spark 3.5 bundles Arrow 12, which fails with `sun.misc.Unsafe ... not available`). This is a Spark limitation, not specific to this library |
| NumPy / pandas / PyArrow | 1.26.4 / 2.2.2 / 15.0.2 | PySpark 3.5 is not compatible with NumPy 2 / pandas 3 |

The package must be importable on the executors (install it on the cluster, or ship it with `spark.sparkContext.addPyFile`).
No `spark-avro` or any other extra JVM package is needed.

## Basic usage

```python
from pyspark.ml.feature import VectorAssembler
from pyspark_isolation_forest import IsolationForest

data = VectorAssembler(inputCols=["x1", "x2", "x3"], outputCol="features").transform(raw)

contamination = 0.1
model = (
    IsolationForest()
    .setNumEstimators(100)
    .setMaxSamples(256)
    .setMaxFeatures(1.0)
    .setBootstrap(False)
    .setContamination(contamination)
    .setContaminationError(0.01 * contamination)
    .setRandomSeed(1)
    .fit(data)
)

result = model.transform(data)            # appends outlierScore and predictedLabel
result.select("features", "outlierScore", "predictedLabel").show()
```

Constructor keywords work as well: `IsolationForest(numEstimators=100, contamination=0.1)`.

**Score semantics** (verified against the Scala source): `outlierScore = 2 ** (-meanPathLength / c(maxSamples))`,
in (0, 1], **higher means more anomalous**. `predictedLabel` is `1.0` if `outlierScore >= threshold`, else `0.0`;
with `contamination = 0.0` no threshold is computed and every label is `0.0`. The threshold is
`approxQuantile(outlierScore, 1 - contamination, contaminationError)` over the training data.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `numEstimators` | `100` | Number of trees (> 0). |
| `maxSamples` | `256.0` | Samples per tree. In (0, 1.0] it is a fraction of the data, > 1.0 a count (floored). The resolved count must be >= 2 and **<= the number of rows** (otherwise an error; it is not clamped). |
| `maxFeatures` | `1.0` | Features per tree. In (0, 1.0] a fraction, > 1.0 a count (floored); each tree draws its own random subset. |
| `contamination` | `0.0` | Expected outlier fraction in [0, 0.5). `0.0` skips the threshold computation. |
| `contaminationError` | `0.0` | Relative error of the quantile used for the threshold, in [0, 1]. `0.0` = exact but slower; `0.01 * contamination` is a good choice for large data. |
| `bootstrap` | `False` | Sample with replacement (Poisson) instead of without (Bernoulli). |
| `randomSeed` | `1` | Seed (> 0). |
| `featuresCol` | `"features"` | Input `Vector` column. |
| `predictionCol` | `"predictedLabel"` | Output label column (appended). |
| `scoreCol` | `"outlierScore"` | Output score column (appended). |
| `extensionLevel` (Extended IF only) | unset | `extensionLevel + 1` non-zero coordinates per hyperplane. `0` = axis-aligned; unset = `numFeatures - 1` of the **per-tree subspace**. Values above that maximum raise an error. |

Invalid values raise `ValueError` at set time (like Spark's `ParamValidators`). The output columns must not already
exist in the input DataFrame.

## Extended Isolation Forest

```python
from pyspark_isolation_forest import ExtendedIsolationForest

model = (
    ExtendedIsolationForest()
    .setNumEstimators(100)
    .setMaxSamples(256)
    .setContamination(0.02)
    .setContaminationError(0.0002)
    .setExtensionLevel(5)          # fully extended for a 6-feature dataset
    .setRandomSeed(1)
    .fit(data)
)
scored = model.transform(data)
```

When `maxFeatures < 1.0`, `extensionLevel` is relative to the feature subspace each tree sees: with 10 features and
`maxFeatures = 0.5` each tree uses 5 features and the valid range is `[0, 4]`. The resolved level is stored on the
fitted model (`model.getExtensionLevel()`); the estimator itself is never modified by `fit`.

## Persistence

```python
model.write().overwrite().save("/path/to/model")           # or model.save(path)
from pyspark_isolation_forest import IsolationForestModel
loaded = IsolationForestModel.load("/path/to/model")
```

Standard Spark ML layout, with **no pickle**: `metadata/` holds JSON (Spark's `DefaultParamsWriter` plus
`outlierScoreThreshold`, `numSamples`, `numFeatures`, `totalNumFeatures`) and `data/` holds Parquet with one row per
tree (flat node arrays). Estimators persist through `DefaultParamsWritable`.
**Models are not compatible with the Scala library's Avro format** (cross-loading was explicitly out of scope).

## Pipelines

```python
from pyspark.ml import Pipeline, PipelineModel

pipeline = Pipeline(stages=[
    VectorAssembler(inputCols=["a", "b", "c"], outputCol="features"),
    IsolationForest(numEstimators=100, contamination=0.02),
])
pm = pipeline.fit(df)
pm.write().overwrite().save("/path/pm")
scored = PipelineModel.load("/path/pm").transform(df)
```

## How it is distributed

1. **Bagging** - `mapInArrow` over the input partitions draws, per row and per tree, a Bernoulli (or Poisson) weight
   with a per-partition seed, over-sampling each tree by 7 sigma exactly as the Scala library does.
2. **Tree training** - `groupBy(treeId).applyInPandas`: one group is one tree.
3. **Collect** - only the finished trees (a few KB each) reach the driver; the dataset is never collected
   (enforced by a test).
4. **Scoring** - a vectorised Arrow `pandas_udf` traverses a broadcast forest for a whole batch at a time (the Scala
   library uses a row-by-row UDF).
5. **Threshold** - Spark's `approxQuantile`, as in Scala.

## Limitations

- Classic PySpark 3.5 only (not tested on Spark Connect or Spark 4).
- JDK 17 (or 8/11) required with Spark 3.5; see the compatibility table.
- Results depend on the number of input partitions (sampling streams are per partition), as in Scala. Re-running with
  the same data, partitioning, seed and Arrow batch size is deterministic.
- Trees are not bit-identical to the Scala library's for the same seed (different RNG; see `COMPATIBILITY.md`).
- Output columns are nullable (the Scala schema marks them non-nullable).
- No ONNX export (upstream supports it for the standard forest only).
- Null feature vectors are not supported.

## Running the tests

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64     # JDK 17
pip install -e ".[test]"
pytest                                                    # ~10-15 minutes on a single core
```

The tests reuse the Scala repo's `mammography.csv` and `shuttle.csv` (ODDS library, Rayana 2016; see upstream `NOTICE`).

## License

BSD 2-Clause, matching upstream. This is an independent port of the algorithms; see upstream for its copyright
notice and the dataset attributions.
