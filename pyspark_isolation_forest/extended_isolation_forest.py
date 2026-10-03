from __future__ import annotations

from pyspark import keyword_only
from pyspark.ml import Estimator
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

from . import training
from .extended_isolation_forest_model import ExtendedIsolationForestModel
from .params import _ExtendedIsolationForestParams
from .utils import validate_and_transform_schema


class ExtendedIsolationForest(Estimator, _ExtendedIsolationForestParams, DefaultParamsReadable,
                              DefaultParamsWritable):
    """Distributed Extended Isolation Forest: random sparse hyperplane splits."""

    @keyword_only
    def __init__(self, *, numEstimators=None, maxSamples=None, maxFeatures=None, contamination=None,
                 contaminationError=None, bootstrap=None, randomSeed=None, featuresCol=None,
                 predictionCol=None, scoreCol=None, extensionLevel=None):
        super().__init__()
        self._init_isolation_forest_defaults()
        self._set(**self._input_kwargs)

    @keyword_only
    def setParams(self, *, numEstimators=None, maxSamples=None, maxFeatures=None, contamination=None,
                  contaminationError=None, bootstrap=None, randomSeed=None, featuresCol=None,
                  predictionCol=None, scoreCol=None, extensionLevel=None):
        return self._set(**self._input_kwargs)

    def _fit(self, dataset):
        validate_and_transform_schema(dataset.schema, self.getFeaturesCol(), self.getPredictionCol(),
                                      self.getScoreCol())
        resolved = training.resolve_from_dataframe(dataset, self.getFeaturesCol(), self.getMaxFeatures(),
                                                   self.getMaxSamples())
        max_level = resolved.num_features - 1
        if self.isSet(self.extensionLevel):
            level = self.getExtensionLevel()
            if level > max_level:
                raise ValueError(f"parameter extensionLevel given invalid value {level}, but must be in "
                                 f"[0, {max_level}] for a subspace of {resolved.num_features} features.")
        else:
            level = max_level
        trees = training.train_trees(dataset, self.getFeaturesCol(), resolved, self.getNumEstimators(),
                                     self.getBootstrap(), self.getRandomSeed(), "extended", level)
        model = ExtendedIsolationForestModel(trees, resolved.num_samples, resolved.num_features,
                                             resolved.total_num_features)
        model._resetUid(self.uid)
        self._copyValues(model)
        model._set(extensionLevel=level)       # resolved level lives on the model; estimator is not mutated
        training.compute_and_set_threshold(model, dataset, self.getScoreCol(), self.getContamination(),
                                           self.getContaminationError())
        return model
