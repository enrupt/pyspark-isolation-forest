"""Distributed training and threshold logic (port of core/SharedTrainLogic.scala + BaggedPoint.scala).

Architecture (same shape as the Scala library, expressed with DataFrame/Arrow primitives):

  1. BAGGING  - `mapInArrow` over the input partitions. Each partition draws, for every row and every
     tree, an independent weight (Binomial(1, rate) without replacement / Poisson(rate) with
     replacement) from an RNG seeded by (randomSeed, partitionId), and emits one record per unit of
     weight keyed by tree id. rate = min((numSamples + 7*sqrt(numSamples)) / N, 1): each tree is
     deliberately over-sampled by 7 sigma and truncated later.
  2. TREES    - `groupBy(treeId).applyInPandas`. One group == one tree (so no hash-partition
     collisions): rows are put in a canonical order, shuffled, truncated to numSamples, a random
     feature subspace is drawn, and the tree is built.
  3. Only the finished trees (tiny) are collected to the driver. The dataset is never collected.

Determinism: for a fixed input partitioning, seed and Arrow batch size the result is deterministic,
including the row order inside each tree group (canonicalised by (partition, sequence) keys, which
is stricter than the Scala implementation whose order depends on shuffle fetch order).
"""
from __future__ import annotations

import logging
from typing import Callable, List

import numpy as np

from .utils import ResolvedParams, resolve_params, sample_fraction

logger = logging.getLogger(__name__)

_BAG_SCHEMA = "treeId int, part int, seq long, f array<float>"
_MAX_WEIGHT_CELLS = 4_000_000  # bound on (rows x trees) drawn at once


def _make_bagging_fn(num_estimators: int, rate: float, bootstrap: bool, seed: int) -> Callable:
    def bag(batches):
        import numpy as _np
        import pyarrow as pa
        from pyspark import TaskContext

        pid = TaskContext.get().partitionId()
        rng = _np.random.default_rng([seed, 0, pid])
        counter = 0
        chunk_rows = max(1, _MAX_WEIGHT_CELLS // max(num_estimators, 1))
        for batch in batches:
            col = batch.column(0)
            n_total = len(col)
            if n_total == 0:
                continue
            flat = col.flatten().to_numpy(zero_copy_only=False)
            if flat.size % n_total != 0:
                raise ValueError("Feature vectors in the input have inconsistent sizes.")
            d = flat.size // n_total
            X = flat.reshape(n_total, d)
            if int(col.value_lengths().to_numpy(zero_copy_only=False).min()) != d:
                raise ValueError("Feature vectors in the input have inconsistent sizes.")
            for start in range(0, n_total, chunk_rows):
                Xc = X[start:start + chunk_rows]
                n = Xc.shape[0]
                if bootstrap:
                    W = rng.poisson(rate, size=(n, num_estimators))
                elif num_estimators == 1 and rate == 1.0:
                    W = _np.ones((n, 1), dtype=_np.int64)       # "no sampling" special case upstream
                else:
                    W = rng.binomial(1, rate, size=(n, num_estimators))
                rows, trees = _np.nonzero(W)
                reps = W[rows, trees]
                rows, trees = _np.repeat(rows, reps), _np.repeat(trees, reps)
                m = rows.size
                if m:
                    values = pa.array(_np.ascontiguousarray(Xc[rows]).reshape(-1), type=pa.float32())
                    offsets = pa.array(_np.arange(0, (m + 1) * d, d, dtype=_np.int32))
                    feats = pa.ListArray.from_arrays(offsets, values)
                    yield pa.RecordBatch.from_arrays(
                        [pa.array(trees.astype(_np.int32)),
                         pa.array(_np.full(m, pid, dtype=_np.int32)),
                         pa.array((counter + start + rows).astype(_np.int64)),
                         feats],
                        names=["treeId", "part", "seq", "f"])
            counter += n_total
    return bag


def _make_tree_fn(tree_kind: str, num_samples: int, num_features: int, seed: int,
                  extension_level: int) -> Callable:
    def train(pdf):
        import numpy as _np
        import pandas as pd
        from pyspark_isolation_forest.tree import IsolationTree
        from pyspark_isolation_forest.extended_tree import ExtendedIsolationTree

        tree_id = int(pdf["treeId"].iloc[0])
        pdf = pdf.sort_values(["part", "seq"], kind="stable")
        X = _np.stack(pdf["f"].to_numpy()).astype(_np.float32)
        rng = _np.random.default_rng([seed, 1, tree_id])
        data = X[rng.permutation(X.shape[0])[:num_samples]]
        if data.shape[0] != num_samples:
            logger.warning("Isolation tree %d is trained using %d data points instead of user specified %d",
                           tree_id, data.shape[0], num_samples)
        feature_indices = _np.sort(rng.permutation(data.shape[1])[:num_features])
        if tree_kind == "standard":
            tree = IsolationTree.fit(data, None, feature_indices, rng=rng)
        else:
            tree = ExtendedIsolationTree.fit(data, None, feature_indices, extension_level, rng=rng)
        row = tree.to_row()
        row["treeId"] = tree_id
        return pd.DataFrame([row])
    return train


def _tree_schema(tree_kind: str) -> str:
    if tree_kind == "standard":
        return ("treeId int, feature array<int>, threshold array<double>, left array<int>, "
                "right array<int>, numInstances array<long>")
    return ("treeId int, k int, indices array<int>, weights array<float>, offset array<double>, "
            "left array<int>, right array<int>, numInstances array<long>")


def features_as_array(df, features_col: str):
    from pyspark.ml.functions import vector_to_array
    return df.select(vector_to_array(df[features_col], "float32").alias("f"))


def resolve_from_dataframe(df, features_col: str, max_features: float, max_samples: float) -> ResolvedParams:
    feats = features_as_array(df, features_col)
    first = feats.head()
    if first is None:
        raise ValueError("Cannot train an isolation forest on an empty dataset.")
    total_features = len(first["f"])
    total_samples = feats.count()
    return resolve_params(max_features, max_samples, total_features, total_samples)


def train_trees(df, features_col: str, resolved: ResolvedParams, num_estimators: int, bootstrap: bool,
                seed: int, tree_kind: str, extension_level: int = 0) -> List:
    """Run the distributed bagging + per-tree training and return the trees ordered by tree id."""
    from .extended_tree import ExtendedIsolationTree
    from .tree import IsolationTree

    rate = sample_fraction(resolved.num_samples, resolved.total_num_samples)
    logger.info("Per-tree sample fraction = %s (numSamples=%d, N=%d)", rate,
                resolved.num_samples, resolved.total_num_samples)
    bagged = features_as_array(df, features_col).mapInArrow(
        _make_bagging_fn(num_estimators, rate, bootstrap, seed), _BAG_SCHEMA)
    trained = bagged.groupBy("treeId").applyInPandas(
        _make_tree_fn(tree_kind, resolved.num_samples, resolved.num_features, seed, extension_level),
        _tree_schema(tree_kind))
    rows = trained.collect()                      # trees only
    by_id = {r["treeId"]: r for r in rows}
    missing = [i for i in range(num_estimators) if i not in by_id]
    if missing:
        raise RuntimeError(
            f"Trees {missing[:5]}{'...' if len(missing) > 5 else ''} received zero samples for tree "
            "training. This can happen with very small maxSamples values. Try increasing maxSamples.")
    cls = IsolationTree if tree_kind == "standard" else ExtendedIsolationTree
    return [cls.from_row(by_id[i].asDict()) for i in range(num_estimators)]


def compute_and_set_threshold(model, df, score_col: str, contamination: float,
                              contaminationError: float) -> None:
    """contamination == 0: no threshold (all labels 0.0). Otherwise approxQuantile(1 - contamination)
    over the training scores; contaminationError == 0.0 makes Spark compute it exactly."""
    if not contamination > 0.0:
        logger.info("Contamination is 0.0: all predicted labels will be 0.0.")
        return
    from pyspark.sql import functions as F

    scores = model.transform(df).select(F.col(score_col).alias("score")).cache()
    try:
        threshold = scores.stat.approxQuantile("score", [1.0 - contamination], contaminationError)[0]
        model._set_outlier_score_threshold(threshold)
        total = scores.count()
        observed = scores.filter(F.col("score") >= threshold).count() / float(total)
        verification_error = contamination * 0.01 if contaminationError == 0.0 else contaminationError
        if abs(observed - contamination) > verification_error:
            logger.warning(
                "Observed contamination is %s, which is outside the expected range of %s +/- %s. If this "
                "is acceptable then it is OK to proceed; for a very large discrepancy retrain with an "
                "exact threshold (contaminationError = 0.0).", observed, contamination, verification_error)
    finally:
        scores.unpersist()
