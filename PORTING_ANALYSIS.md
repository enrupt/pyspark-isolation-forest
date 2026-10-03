# PORTING_ANALYSIS.md

Analysis of `linkedin/isolation-forest` (master @ `171eaf5`, Scala, Spark 3.5.5) for a native PySpark port.
Everything below is taken from the **source and tests**, not the README. Where README and source differ, the source is followed (section 6).

Files read: all of `isolation-forest/src/main` (19 files, ~3.1k lines) and the Scala tests (`IsolationForestTest`, `IsolationTreeTest`,
`ExtendedIsolationForestTest`, `ExtendedIsolationTreeTest`, `*ModelWriteReadTest`, `BaggedPointTest`, `UtilsTest`, `TestUtils`), plus the saved model fixtures.

---

## 1. Scala class -> Python class mapping

| Scala | Role | Python |
|---|---|---|
| `IsolationForest` (Estimator, `DefaultParamsWritable`) | standard estimator | `IsolationForest` |
| `IsolationForestModel` (Model, `MLWritable`) | standard model | `IsolationForestModel` |
| `ExtendedIsolationForest` / `ExtendedIsolationForestModel` | EIF estimator / model | same names |
| `IsolationForestParamsBase` | 10 shared params + defaults | `params.py` (`_IsolationForestParams`) |
| `ExtendedIsolationForestParams` | adds `extensionLevel` | `_ExtendedIsolationForestParams` |
| `IsolationTree` + `Nodes.{InternalNode,ExternalNode}` | standard tree | `tree.py` |
| `ExtendedIsolationTree` + `ExtendedNodes` + `ExtendedUtils.SplitHyperplane` | EIF tree, sparse hyperplane | `extended_tree.py` |
| `core.Utils` (`avgPathLength`, schema validation, `DataPoint`) | shared helpers | `utils.py` |
| `core.SharedTrainLogic` + `core.BaggedPoint` | param resolution, bagging, tree training, threshold | `training.py` |
| `*ModelReadWrite`, `IsolationForestModelReadWriteUtils` | Avro tree storage + JSON metadata | `persistence.py` (Parquet + JSON) |
| `isolation-forest-onnx` | ONNX export | out of scope |

Note: `Param` lives in `pyspark.ml.param` (not `.shared`). PySpark has no `LongParam`, so `randomSeed` uses `TypeConverters.toInt`.

## 2. Public API, defaults and validators (verified in `IsolationForestParamsBase`)

| Param | Type | Default | Validator |
|---|---|---|---|
| numEstimators | Int | 100 | > 0 |
| maxSamples | Double | 256.0 | > 0 |
| contamination | Double | 0.0 | [0.0, 0.5) |
| contaminationError | Double | 0.0 | [0.0, 1.0] |
| maxFeatures | Double | 1.0 | > 0 |
| bootstrap | Boolean | false | - |
| randomSeed | Long | 1 | > 0 (so 0 is **invalid**) |
| featuresCol / predictionCol / scoreCol | String | `features` / `predictedLabel` / `outlierScore` | - |
| extensionLevel (EIF only) | Int | **unset** | >= 0 |

All defaults in the brief/README match the source. Model-only state: `numSamples`, `numFeatures`, `totalNumFeatures`, `outlierScoreThreshold` (sentinel `-1` = none).

## 3. Algorithm components (as implemented upstream)

### 3.1 Parameter resolution (`validateAndResolveParams`)
- `numFeatures = floor(maxFeatures)` if `maxFeatures > 1.0` else `floor(maxFeatures * totalFeatures)`; must be in `[1, totalFeatures]`.
- `numSamples = floor(maxSamples)` if `maxSamples > 1.0` else `floor(maxSamples * N)`; must be `>= 2` **and `<= N`**. So `maxSamples > N` is an **error** (not clamped), and `maxSamples = 1.0` means 100% (fraction).

### 3.2 Distributed training pipeline (`SharedTrainLogic`)
1. Features are cast to **`Float`** (`DataPoint(Array[Float])`): data in float32, split values in float64.
2. **Bagging** (`BaggedPoint.convertToBaggedRDD`): per row and per tree, draw a weight. Per input partition `p` the RNG is seeded `randomSeed + p` (commons-math3 `Binomial(1, rate)`, or `Poisson(rate)` if `bootstrap`). `rate = min((numSamples + 7*sqrt(numSamples)) / N, 1.0)` - deliberately **over-samples by 7 sigma**. Special case: no sampling when `numEstimators == 1 && rate == 1.0`.
3. **Flatten**: emit `(treeId, row)` once per unit weight.
4. **Shuffle to trees**: `partitionBy(HashPartitioner(numEstimators))` - **one tree per partition**.
5. **Per-partition tree training**: seed `= randomSeed + 2*(numInputPartitions+1) + partitionId`; `Random(seed).shuffle(rows).slice(0, numSamples)`; random feature subset from the same stream; tree built with a fresh `Random(seed)`.
6. Only the finished trees are `collect()`ed to the driver; the data is never collected.

Consequences: results depend on the **number of input partitions**; sampling is Bernoulli per (row, tree), *not* an exact "numSamples without replacement" - each tree gets a random count, then is truncated after a shuffle. An empty tree partition throws; a short one only warns.

### 3.3 Standard tree (`IsolationTree`)
- `heightLimit = ceil(log10(n)/log10(2))` (identical to exact `ceil(log2 n)` for all n < 70,000, but the formula is reproduced).
- Per node: candidate features are tried in random order without replacement; the first with min != max is used; `splitValue = (max-min)*U[0,1) + min`. **Constant features are skipped and retried.**
- Leaf if no splittable feature, `depth >= heightLimit`, or `n <= 1` -> `ExternalNode(n)` (`n > 0` enforced). Split: `x < splitValue` -> left. Path length = depth + `c(n_leaf)`.
- Quirk: the feature search (and its RNG draws) happens before the leaf test.

### 3.4 Extended tree (`ExtendedIsolationTree`)
- Same height limit and leaf rule, but **no retry on constant features** and **empty children allowed** (`ExtendedExternalNode(0)`, `c(0) = 0`).
- Per node: `nNonZero = min(extensionLevel+1, dim)`; coordinates chosen by shuffling the subspace; Gaussian weights, L2-normalised, stored as **Float**. Intercept per coordinate uniform in [min, max]; `offset = sum(w_float * intercept)` in double. Split `dot(x) < offset` -> left; `dot` multiplies in float32 and accumulates in float64.
- `extensionLevel` is resolved at fit time: unset -> `numFeatures-1` of the **resolved subspace**; an explicit value above that throws. The estimator is not mutated (there is a regression test upstream).
- Confirms the README: `extensionLevel` is relative to the `maxFeatures` subspace, and EIF level 0 is **not** equivalent to standard IF.

### 3.5 Scoring (`transform`)
- `score = 2^(-meanPathLength / c(numSamples))` using the **resolved `numSamples`**; float32 path sums, `Math.pow` in double. Higher = more anomalous; range (0, 1].
- `c(n) = 0` for `n <= 1`, else `2*(ln(n-1) + 0.5772156649f) - 2*(n-1)/n` in float32 (`c(2)=0.15443134`, `c(10)=3.7488806`).
- `predictedLabel = 1.0 if score >= threshold else 0.0` when `threshold > 0`; otherwise all `0.0`.
- Output columns must not already exist; both are non-nullable doubles; feature size is validated against `totalNumFeatures`.
- Upstream scoring is itself a row-wise UDF over a broadcast of the trees.

### 3.6 Contamination threshold (`computeAndSetModelThreshold`)
- `contamination == 0` -> skipped (no scoring pass at fit time).
- Else: score the training DataFrame, then `approxQuantile("score", [1 - contamination], contaminationError)` (Greenwald-Khanna; `contaminationError = 0.0` makes Spark compute it exactly). PySpark exposes the same function, so the step is reproducible exactly given identical scores.
- Then recount the observed fraction with `score >= threshold`; warn if outside `contamination +/- error` (error = 1% of contamination when `contaminationError == 0`).

### 3.7 Persistence
- `metadata/part-00000`: Spark `DefaultParamsWriter` JSON + `outlierScoreThreshold`, `numSamples`, `numFeatures`, `totalNumFeatures`.
- `data/`: **Avro**, one row per node, pre-order ids from 0 (EIF adds `indices`, `weights`, `offset`). Two Scala-written fixtures exist in `src/test/resources`.

## 4. Spark-specific components
`Estimator`/`Model`, `Params`, `DefaultParamsWritable`/`MLWritable`, `Pipeline` compatibility, broadcast of trees, RDD `mapPartitionsWithIndex`/`partitionBy`/`TaskContext`, `approxQuantile`, `udf`, Avro data source.

## 5. Reproducibility matrix

| Component | Exactly reproducible? | Notes |
|---|---|---|
| Defaults, validators, param resolution, error cases | **Yes** | pure logic |
| `c(n)`, score formula, float32 semantics, `>=` threshold rule | **Yes** | `np.float32` |
| Height limit, leaf rules, split inequality, path length | **Yes** | |
| Tree structure given identical subset/seed/features | Only with a bit-exact port of `java.util.Random` + Scala `shuffle` | not done (see section 11) |
| Contamination threshold given the same scores | **Yes** | same `approxQuantile` |
| Per-(row,tree) sampling | **No** (statistically equivalent) | commons-math3 Well19937c vs NumPy |
| Row order inside a tree partition | **No - not even Scala-to-Scala guaranteed** | shuffle fetch order; the port canonicalises it |
| End-to-end identical forests | **No** | consequence of the above; also depends on partition count |

## 6. Discrepancies between the brief / README and the source

1. **"Sampling without replacement"**: Bernoulli per (row, tree) with 7-sigma over-sampling, then shuffle + truncate; trees can get fewer than `numSamples` rows (warning).
2. **"Leaf size"**: there is no min-leaf parameter; leaves are `n <= 1`, height limit, or no splittable feature.
3. **`maxSamples > N`** is an error, not clamped; `maxSamples` in (1, 2) fails (`floor < 2`).
4. **Score**: `2^(-E[h]/c(numSamples))`, higher = more anomalous, threshold test is `>=`, a missing threshold yields all-0 predictions (not a fixed 0.5).
5. **`randomSeed = 0` is rejected.**
6. README shows `maxSamples = 256`; the param is really `256.0`.
7. **Scoring is not "avoiding UDFs" upstream**: Scala also uses a row-wise UDF; a vectorised path is an improvement, not parity.
8. ONNX export exists upstream only for standard IF; not ported.

## 7. Python architecture (preserves the distributed design, collects only trees)

- **Bagging**: `mapInArrow` over the input partitions, per-partition seed, vectorised Binomial/Poisson draws, output `(treeId, part, seq, features)`.
- **Trees**: `groupBy("treeId").applyInPandas` - one group is one tree; canonical row order, shuffle/slice, feature subset, build; only tree arrays are collected.
- **Transform**: Arrow `pandas_udf` over a broadcast forest with numpy-vectorised traversal.
- **Threshold**: `df.stat.approxQuantile`.
- **Persistence**: custom `MLWriter`/`MLReader` using `DefaultParamsWriter` metadata; no pickle.

## 8. Test port plan

| Scala test | Python |
|---|---|
| `IsolationForestTest` (mammography AUROC 0.86 +/- 0.02, exact contamination, zero contamination, shuttle, invalid `maxSamples`) | `test_isolation_forest.py`, with the repo's own CSVs |
| `IsolationTreeTest`, `ExtendedIsolationTreeTest` | `test_tree.py`, `test_extended_tree.py` |
| `BaggedPointTest` | `test_bagging.py` (distribution checks) |
| `UtilsTest` | `test_utils.py` (exact float32 expectations) |
| `*ModelWriteReadTest` | `test_persistence.py` (own format) |
| EIF default-extension-level test | `test_extended_isolation_forest.py` |
| ONNX integration | not ported |

## 9. Environment and validation feasibility (verified in the sandbox)

- PySpark 3.5.5 installs; JDK 21 is the default but **Spark 3.5 + JDK 21 fails for every Arrow UDF** (bundled Arrow 12), so JDK 17 was installed.
- `github.com` works; Maven Central and Gradle hosts return 403, so the Scala library could not be built here.
- 1 CPU core, 3 GB RAM: 1M-row benchmarks are possible but say little about cluster behaviour.

## 10. Decisions that were open at analysis time

1. RNG for tree construction: bit-exact `java.util.Random` port vs NumPy.
2. Persistence: Scala-compatible Avro vs Spark-ML Parquet.

## 11. Outcome (added after implementation)

Scala compatibility was dropped as a requirement, which resolved both decisions: **(1)** NumPy `Generator` (statistical parity, not bit-exact trees);
**(2)** JSON metadata + Parquet, not the Scala Avro layout. Scala-side compilation/validation was not performed; fidelity was established by independent
re-implementation of the Scala rules and by the Scala repo's own datasets and tolerances. See `COMPATIBILITY.md`.
