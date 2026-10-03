# BENCHMARKS.md

Python port only. **No Scala timings exist**: Maven/Gradle hosts were unreachable in the benchmarking sandbox, so the upstream
library could not be built, and a Scala comparison was out of scope. Treat these numbers as a characterisation of this
implementation, not a Scala-vs-Python comparison.

**Setup:** 1 CPU core, 3 GB RAM, Spark 3.5.5 `local[2]`, JDK 17, Python 3.12, NumPy 1.26.4. 10 features, 100 trees,
`maxSamples=256`, data generated inside Spark (`randn`), cached before timing, 8 partitions. Single run per cell, no repeats:
**expect roughly +/-20% noise**, and the very first cell (IsolationForest, 10k) includes JVM/Python-worker warm-up.
Cluster behaviour (shuffle, network, many executors) is *not* captured by a single-machine local run.

"fit" = `contamination=0.0` (no threshold pass). "fit + contamination" = `contamination=0.01, contaminationError=1e-4`
(adds a scoring pass over the training data plus `approxQuantile` and a verification count).
"transform" = score all rows and aggregate (forces full evaluation).

| Model | Rows | fit (s) | fit + contamination (s) | transform (s) | jobs / stages (fit) | jobs / stages (fit + cont.) |
|---|---:|---:|---:|---:|---:|---:|
| IsolationForest | 10,000 | 7.55* | 6.88 | 2.89 | 5 / 12 | 11 / 27 |
| ExtendedIsolationForest | 10,000 | 3.93 | 7.80 | 3.10 | 5 / 12 | 11 / 27 |
| IsolationForest | 100,000 | 3.49 | 8.09 | 3.75 | 5 / 12 | 11 / 27 |
| ExtendedIsolationForest | 100,000 | 3.74 | 17.16 | 12.85 | 5 / 12 | 11 / 27 |
| IsolationForest | 1,000,000 | 11.05 | 34.32 | 25.87 | 5 / 12 | 11 / 27 |
| ExtendedIsolationForest | 1,000,000 | 6.38 | 116.08 | 110.80 | 5 / 12 | 11 / 27 |

\* includes warm-up.

### Memory (peak RSS, local mode)

| Rows | Driver JVM (also executor in local mode) | Python workers (sum) | Driver Python process |
|---:|---:|---:|---:|
| 10,000 | ~450 MB | ~320 MB | ~125 MB |
| 100,000 | ~490 MB | ~340 MB | ~128 MB |
| 1,000,000 | ~815 MB | ~350 MB | ~128 MB |

The JVM growth at 1M rows is the cached input DataFrame. **Driver Python memory and Python-worker memory do not grow with N**,
which is the intended consequence of not collecting the data (and a test enforces that only the trees are collected).
Executor memory in a real cluster is not measured here.

## What the numbers say

1. **Fit cost is almost independent of the number of rows.** Each tree trains on ~370 sampled rows, so fit is dominated by
   building 100 small trees plus one pass over the data for bagging: 1M rows fit in 6-11 s on one core.
2. **Job/stage count is constant** (5 jobs / 12 stages for a plain fit; 11 / 27 with a contamination threshold) and does not
   depend on N or the model type.
3. **The threshold computation is the expensive part of fit at scale**, as upstream warns: with `contamination > 0` fit time at
   1M rows is 3x (standard) to 18x (extended) the plain fit, because it scores the whole training set. Using
   `contaminationError > 0` helps only the quantile step, not the scoring pass.
4. **Transform scales linearly with N**, ~26 us/row for the standard forest and ~111 us/row for the extended forest on one core.

## Bottleneck: Extended IF scoring

Extended scoring is ~4.3x slower than standard at 1M rows (111 s vs 26 s). Cause: every node visit evaluates a sparse dot
product over `k = extensionLevel + 1 = 10` coordinates (default fully-extended with 10 features) via a fancy-indexed
gather of an `(m, 10)` float32 block, versus a single comparison for the standard forest. The traversal is level-synchronous
NumPy, so cost ~ rows x trees x depth x k with Python-level overhead per (tree, level).

Not done (deliberately, per "do not optimise prematurely"): lower `extensionLevel` (cost scales with `k`), batching several trees
per NumPy call, or a compiled kernel (Numba/C). These are the first things to try if scoring throughput matters. Scoring is
embarrassingly parallel, so it also scales with executor count in a way this single-core run cannot show.

## Reproduce

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
python benchmarks/benchmark.py --rows 10000 100000 1000000 --features 10 --trees 100 --out results.json
```
