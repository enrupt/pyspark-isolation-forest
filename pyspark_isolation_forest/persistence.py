"""Spark ML persistence for the forest models: no pickle, no JVM model classes.

Layout (Spark ML conventions):
    <path>/metadata/   JSON written by pyspark's DefaultParamsWriter (+ model fields below)
    <path>/data/       Parquet, one row per tree, with flat node arrays

Extra metadata: outlierScoreThreshold, numSamples, numFeatures, totalNumFeatures, treeKind.
"""
from __future__ import annotations

from pyspark.ml.util import DefaultParamsReader, DefaultParamsWriter, MLReader, MLWriter

from .extended_tree import ExtendedIsolationTree
from .tree import IsolationTree

_STANDARD_SCHEMA = ("treeID int, feature array<int>, threshold array<double>, left array<int>, "
                    "right array<int>, numInstances array<long>")
_EXTENDED_SCHEMA = ("treeID int, k int, indices array<int>, weights array<float>, offset array<double>, "
                    "left array<int>, right array<int>, numInstances array<long>")


class ForestModelWriter(MLWriter):
    def __init__(self, instance):
        super().__init__()
        self.instance = instance

    def saveImpl(self, path: str) -> None:
        m = self.instance
        extra = {"outlierScoreThreshold": float(m.getOutlierScoreThreshold()),
                 "numSamples": int(m.getNumSamples()), "numFeatures": int(m.getNumFeatures()),
                 "totalNumFeatures": int(m.getTotalNumFeatures()), "treeKind": m._TREE_KIND,
                 "numTrees": len(m._trees)}
        DefaultParamsWriter.saveMetadata(m, path, self.sc, extraMetadata=extra)
        schema = _STANDARD_SCHEMA if m._TREE_KIND == "standard" else _EXTENDED_SCHEMA
        rows = [dict(t.to_row(), treeID=i) for i, t in enumerate(m._trees)]
        self.sparkSession.createDataFrame(rows, schema).repartition(1).write.parquet(f"{path}/data")


class ForestModelReader(MLReader):
    def __init__(self, cls):
        super().__init__()
        self.cls = cls

    def load(self, path: str):
        meta = DefaultParamsReader.loadMetadata(path, self.sc)
        kind = meta.get("treeKind")
        if kind != self.cls._TREE_KIND:
            raise ValueError(f"Model at {path} has treeKind={kind!r}, expected {self.cls._TREE_KIND!r}.")
        rows = sorted(self.sparkSession.read.parquet(f"{path}/data").collect(), key=lambda r: r["treeID"])
        if len(rows) != meta["numTrees"]:
            raise ValueError(f"Expected {meta['numTrees']} trees at {path}, found {len(rows)}.")
        tree_cls = IsolationTree if kind == "standard" else ExtendedIsolationTree
        trees = [tree_cls.from_row(r.asDict()) for r in rows]
        model = self.cls(trees, meta["numSamples"], meta["numFeatures"], meta["totalNumFeatures"])
        model._resetUid(meta["uid"])
        DefaultParamsReader.getAndSetParams(model, meta)
        model._set_outlier_score_threshold(meta["outlierScoreThreshold"])
        return model
