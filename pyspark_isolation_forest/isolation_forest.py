from __future__ import annotations

from pyspark import keyword_only
from pyspark.ml import Estimator
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

from . import training
from .isolation_forest_model import IsolationForestModel
from .params import _IsolationForestParams
from .utils import validate_and_transform_schema


class IsolationForest(Estimator, _IsolationForestParams, DefaultParamsReadable, DefaultParamsWritable):
    """Distributed standard Isolation Forest (native PySpark port of linkedin/isolation-forest)."""

    @keyword_only
    def __init__(self, *, numEstimators=None, maxSamples=None, maxFeatures=None, contamination=None,
                 contaminationError=None, bootstrap=None, randomSeed=None, featuresCol=None,
                 predictionCol=None, scoreCol=None):
        super().__init__()
        self._init_isolation_forest_defaults()
        self._set(**self._input_kwargs)

    @keyword_only
    def setParams(self, *, numEstimators=None, maxSamples=None, maxFeatures=None, contamination=None,
                  contaminationError=None, bootstrap=None, randomSeed=None, featuresCol=None,
                  predictionCol=None, scoreCol=None):
        return self._set(**self._input_kwargs)

    def _fit(self, dataset):
        validate_and_transform_schema(dataset.schema, self.getFeaturesCol(), self.getPredictionCol(),
                                      self.getScoreCol())
        resolved = training.resolve_from_dataframe(dataset, self.getFeaturesCol(), self.getMaxFeatures(),
                                                   self.getMaxSamples())
        trees = training.train_trees(dataset, self.getFeaturesCol(), resolved, self.getNumEstimators(),
                                     self.getBootstrap(), self.getRandomSeed(), "standard")
        model = IsolationForestModel(trees, resolved.num_samples, resolved.num_features,
                                     resolved.total_num_features)
        model._resetUid(self.uid)
        self._copyValues(model)
        training.compute_and_set_threshold(model, dataset, self.getScoreCol(), self.getContamination(),
                                           self.getContaminationError())
        return model
