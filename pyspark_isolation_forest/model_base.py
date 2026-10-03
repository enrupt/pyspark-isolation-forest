"""Behaviour shared by IsolationForestModel and ExtendedIsolationForestModel (transform, threshold)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pyspark.ml import Model
from pyspark.ml.util import MLReadable, MLWritable

from .persistence import ForestModelReader, ForestModelWriter
from .utils import scores_from_path_lengths, validate_and_transform_schema

UNSET_THRESHOLD = -1.0


class _ForestModelBase(Model, MLReadable, MLWritable):
    _TREE_KIND = ""

    def __init__(self, trees=None, numSamples=None, numFeatures=None, totalNumFeatures=None):
        super().__init__()
        self._init_isolation_forest_defaults()
        trees = list(trees) if trees is not None else []
        if numSamples is not None and numSamples <= 0:
            raise ValueError(f"parameter numSamples must be >0, but given invalid value {numSamples}")
        if numFeatures is not None and numFeatures <= 0:
            raise ValueError(f"parameter numFeatures must be >0, but given invalid value {numFeatures}")
        if totalNumFeatures is not None and numFeatures is not None and numFeatures > totalNumFeatures:
            raise ValueError("parameter numFeatures must be <= totalNumFeatures")
        self._trees = trees
        self._num_samples = numSamples
        self._num_features = numFeatures
        self._total_num_features = totalNumFeatures
        self._outlier_score_threshold = UNSET_THRESHOLD

    # ---- model state -------------------------------------------------------------------
    @property
    def trees(self):
        return list(self._trees)

    def getNumSamples(self): return self._num_samples
    def getNumFeatures(self): return self._num_features
    def getTotalNumFeatures(self): return self._total_num_features
    def getOutlierScoreThreshold(self): return self._outlier_score_threshold

    def _set_outlier_score_threshold(self, value: float) -> None:
        if not (value == UNSET_THRESHOLD or 0.0 <= value <= 1.0):
            raise ValueError("parameter outlierScoreThreshold must be equal to -1 (no threshold) or be in "
                             f"the range [0, 1], but given invalid value {value}")
        self._outlier_score_threshold = float(value)

    # ---- scoring -----------------------------------------------------------------------
    def _score_matrix(self, X: np.ndarray) -> np.ndarray:
        """Scores for an (n, d) float32 matrix: 2 ** (-mean path / c(numSamples))."""
        total = self._total_num_features
        if total is not None and X.shape[1] != total:
            raise ValueError(f"Input feature vector size {X.shape[1]} did not match the model's training "
                             f"dimension {total}.")
        path_sum = np.zeros(X.shape[0], dtype=np.float32)
        for tree in self._trees:                       # sequential float32 accumulation, as upstream
            path_sum = path_sum + tree.path_length(X)
        return scores_from_path_lengths(path_sum, len(self._trees), self._num_samples)

    def _transform(self, dataset):
        from pyspark.ml.functions import vector_to_array
        from pyspark.sql import functions as F
        from pyspark.sql.functions import pandas_udf

        if self._num_samples is None or self._num_samples < 2:
            raise ValueError(f"Cannot score with numSamples={self._num_samples}; expected numSamples >= 2.")
        if not self._trees:
            raise ValueError("Cannot score with an empty model.")
        features_col, score_col, pred_col = self.getFeaturesCol(), self.getScoreCol(), self.getPredictionCol()
        validate_and_transform_schema(dataset.schema, features_col, pred_col, score_col)

        bc = dataset.sparkSession.sparkContext.broadcast(self)  # trees/params only; see __getstate__

        @pandas_udf("double")
        def score_udf(features: pd.Series) -> pd.Series:
            if len(features) == 0:
                return pd.Series([], dtype="float64")
            X = np.stack(features.to_numpy()).astype(np.float32)
            return pd.Series(bc.value._score_matrix(X))

        scored = dataset.withColumn(score_col, score_udf(vector_to_array(F.col(features_col), "float32")))
        if self._outlier_score_threshold > 0:
            return scored.withColumn(pred_col, (F.col(score_col) >= self._outlier_score_threshold).cast("double"))
        return scored.withColumn(pred_col, F.lit(0.0))

    # ---- persistence -------------------------------------------------------------------
    def write(self):
        return ForestModelWriter(self)

    @classmethod
    def read(cls):
        return ForestModelReader(cls)
