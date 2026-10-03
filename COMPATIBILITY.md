# COMPATIBILITY.md

How this PySpark port relates to `linkedin/isolation-forest` (Scala, master @ `171eaf5`).
Source of truth was the Scala **source and tests**; see `PORTING_ANALYSIS.md` for the full reading.

**Scope note:** cross-loading of Scala-saved models and a side-by-side Scala-vs-Python run were explicitly dropped
(not required). Fidelity is instead established by (a) independent re-implementation of the Scala rules, (b) the
Scala repo's own test datasets and tolerances, and (c) statistical checks. No Scala code was executed.

## 1. Behaviour reproduced exactly (and how it is tested)

| Behaviour | Test |
|---|---|
| All parameter names, defaults and validators (`randomSeed > 0`, `contamination in [0, 0.5)`, etc.) | `test_isolation_forest.py::test_defaults_*`, `test_invalid_params_*` |
| `maxSamples` / `maxFeatures` resolution: fraction if `<= 1.0`, floored count if `> 1.0`; `numSamples >= 2`; **`maxSamples > N` is an error** | `test_utils.py::TestResolveParams`, `test_max_samples_*` |
| `c(n)` in float32, including Scala's own `UtilsTest` values (`c(2)=0.15443134`, `c(10)=3.7488806`, `c(Long.MaxValue)=86.49098`) | `test_utils.py` |
| Height limit `ceil(log10(n)/log10(2))` | `test_utils.py` (equals exact `ceil(log2 n)` for all n < 20,000) |
| Float32 feature data; `x < splitValue` goes left; comparison against a float64 split value | `test_tree.py` |
| Standard tree: random feature order, **constant features skipped and retried**, leaf if no splittable feature / height limit / `n <= 1` | `test_tree.py`, `test_compatibility.py` |
| Extended tree: sparse L2-normalised float32 hyperplane, float32 products accumulated in float64, float64 offset, **no retry**, empty children allowed (`c(0)=0`) | `test_extended_tree.py` |
| `extensionLevel` unset -> `numFeatures-1` of the per-tree subspace; explicit value above it is an error; estimator not mutated by `fit` | `test_extended_isolation_forest.py` |
| Score `2 ** (-mean path / c(numSamples))` using the *resolved* `numSamples`; float32 path accumulation | `test_compatibility.py::test_spark_scores_equal_independent_scala_style_reference` (match to 1e-12) |
| Label rule `score >= threshold`; no threshold when `contamination == 0` (all labels `0.0`) | `test_compatibility.py`, `test_isolation_forest.py` |
| Threshold = `approxQuantile(score, 1 - contamination, contaminationError)` over training scores (exact when error is 0) | `test_compatibility.py::test_threshold_is_the_exact_*` |
| Per-tree sampling rate `min((numSamples + 7*sqrt(numSamples)) / N, 1)`; Bernoulli without / Poisson with replacement; the `numEstimators == 1 && rate == 1` no-sampling case | `test_utils.py`, `test_bagging.py` |
| Output schema: score column then prediction column appended; output columns must not pre-exist; feature dimension validated at transform | `test_isolation_forest.py` |
| Accuracy on the Scala tests' datasets with the Scala tests' tolerances (mammography AUROC 0.86 +/- 0.02; shuttle means 0.61 / 0.41 +/- 0.02, AUROC > 0.99; exact-contamination case) | `test_isolation_forest.py` |

## 2. Behaviour that differs (and why)

| Difference | Cause | Impact |
|---|---|---|
| **Trees are not bit-identical to Scala's for the same seed** | NumPy `Generator` (PCG64) instead of `java.util.Random` / commons-math3 Well19937c | Statistically equivalent; scores differ slightly run-to-run across implementations. Same Python implementation + seed + partitioning + Arrow batch size is deterministic; a different seed gives a different model |
| Row order inside a tree's sample is canonicalised by `(partition, sequence)` | Scala's order depends on shuffle fetch order, so it is not even Scala-to-Scala reproducible | Python is *more* deterministic than upstream |
| Per-tree RNG seeds are `(randomSeed, treeId)`; bagging seeds are `(randomSeed, partitionId)` | Scala uses `randomSeed + 2*(numPartitions+1) + partitionId` | Tree stage is independent of input partition count; sampling still depends on it (as in Scala) |
| Tree stage uses `groupBy(treeId).applyInPandas` | Scala uses `partitionBy(HashPartitioner)` + `mapPartitions` | Same one-tree-per-group architecture; no hash collisions possible |
| Feature search happens *after* the cheap leaf tests in the standard tree | Scala draws random numbers for the feature search even at leaves | Different RNG consumption only; no behavioural effect |
| A uniform draw of exactly 0.0 (split == min, empty left child) yields a leaf | Upstream would throw (`ExternalNode(0)` is invalid) | Probability ~2^-53 per node |
| Scoring is a vectorised Arrow `pandas_udf` | Scala uses a row-by-row UDF | Same results (checked to 1e-12); different performance |
| Persistence: JSON metadata + **Parquet**, one row per tree | Scala uses Avro one row per node | **Not cross-loadable** with the Scala library |
| Output columns are nullable | `pandas_udf` + `withColumn` | Scala marks them non-nullable |
| Metadata `class` is the Python class path | Spark records the implementing class | Scala loader would reject it |
| Python `ValueError` instead of `IllegalArgumentException`; parameter validators fire at *set* time | language convention | Same conditions are rejected |
| Spark Connect, Spark 4 untested; JDK 21 unsupported with Spark 3.5 | Arrow 12 bundled with Spark 3.5 | See README |

## 3. Not supported

- Loading/saving the Scala Avro model format; ONNX export (`isolation-forest-onnx`).
- Null feature vectors.
- Spark Connect (the library relies on classic `SparkContext` for persistence/broadcast).

## 4. Performance

See `BENCHMARKS.md`. No Scala timings exist: the Scala library could not be built in the benchmarking sandbox (Maven/Gradle
hosts blocked) and a Scala comparison was out of scope.
