"""Spark ML params shared by the standard and extended isolation forests.

Names, defaults and validators mirror IsolationForestParamsBase.scala / ExtendedIsolationForestParams.scala.
"""
from __future__ import annotations

from pyspark.ml.param import Param, Params, TypeConverters

_VALIDATORS = {
    "numEstimators": (lambda v: v > 0, "numEstimators must be > 0"),
    "maxSamples": (lambda v: v > 0, "maxSamples must be > 0"),
    "contamination": (lambda v: 0.0 <= v < 0.5, "contamination must be in [0.0, 0.5)"),
    "contaminationError": (lambda v: 0.0 <= v <= 1.0, "contaminationError must be in [0.0, 1.0]"),
    "maxFeatures": (lambda v: v > 0, "maxFeatures must be > 0"),
    "randomSeed": (lambda v: v > 0, "randomSeed must be > 0"),
    "extensionLevel": (lambda v: v >= 0, "extensionLevel must be >= 0"),
}


class _IsolationForestParams(Params):
    numEstimators = Param(Params._dummy(), "numEstimators", "The number of trees in the ensemble.",
                          typeConverter=TypeConverters.toInt)
    maxSamples = Param(
        Params._dummy(), "maxSamples",
        "The number of samples used to train each tree. If between 0.0 and 1.0 it is treated as a "
        "fraction; if > 1.0 it is treated as a count.", typeConverter=TypeConverters.toFloat)
    contamination = Param(
        Params._dummy(), "contamination",
        "The fraction of outliers in the training data set. If 0.0, training is faster and all "
        "predicted labels are 0.0; the model and scores are otherwise unaffected.",
        typeConverter=TypeConverters.toFloat)
    contaminationError = Param(
        Params._dummy(), "contaminationError",
        "Error allowed when computing the contamination threshold. 0.0 forces an exact (slow) "
        "calculation.", typeConverter=TypeConverters.toFloat)
    maxFeatures = Param(
        Params._dummy(), "maxFeatures",
        "The number of features used to train each tree. If between 0.0 and 1.0 it is treated as "
        "a fraction; if > 1.0 it is treated as a count.", typeConverter=TypeConverters.toFloat)
    bootstrap = Param(Params._dummy(), "bootstrap",
                      "If true, draw the sample for each tree with replacement.",
                      typeConverter=TypeConverters.toBoolean)
    randomSeed = Param(Params._dummy(), "randomSeed", "The seed used for the random number generator.",
                       typeConverter=TypeConverters.toInt)
    featuresCol = Param(Params._dummy(), "featuresCol", "The feature vector.",
                        typeConverter=TypeConverters.toString)
    predictionCol = Param(Params._dummy(), "predictionCol", "The predicted label.",
                          typeConverter=TypeConverters.toString)
    scoreCol = Param(Params._dummy(), "scoreCol", "The outlier score.",
                     typeConverter=TypeConverters.toString)

    def _init_isolation_forest_defaults(self):
        self._setDefault(
            numEstimators=100, maxSamples=256.0, contamination=0.0, contaminationError=0.0,
            maxFeatures=1.0, bootstrap=False, randomSeed=1, featuresCol="features",
            predictionCol="predictedLabel", scoreCol="outlierScore")

    # -- validation: like Spark's ParamValidators, but raised at set time --------------------
    def _set(self, **kwargs):
        for name, value in kwargs.items():
            if value is None or name not in _VALIDATORS:
                continue
            converted = getattr(self, name).typeConverter(value)
            check, message = _VALIDATORS[name]
            if not check(converted):
                raise ValueError(f"{message}, but given invalid value {value!r}")
        return super()._set(**kwargs)

    # -- fluent setters / getters -------------------------------------------------------------
    def setNumEstimators(self, value): return self._set(numEstimators=value)
    def getNumEstimators(self): return self.getOrDefault(self.numEstimators)
    def setMaxSamples(self, value): return self._set(maxSamples=value)
    def getMaxSamples(self): return self.getOrDefault(self.maxSamples)
    def setContamination(self, value): return self._set(contamination=value)
    def getContamination(self): return self.getOrDefault(self.contamination)
    def setContaminationError(self, value): return self._set(contaminationError=value)
    def getContaminationError(self): return self.getOrDefault(self.contaminationError)
    def setMaxFeatures(self, value): return self._set(maxFeatures=value)
    def getMaxFeatures(self): return self.getOrDefault(self.maxFeatures)
    def setBootstrap(self, value): return self._set(bootstrap=value)
    def getBootstrap(self): return self.getOrDefault(self.bootstrap)
    def setRandomSeed(self, value): return self._set(randomSeed=value)
    def getRandomSeed(self): return self.getOrDefault(self.randomSeed)
    def setFeaturesCol(self, value): return self._set(featuresCol=value)
    def getFeaturesCol(self): return self.getOrDefault(self.featuresCol)
    def setPredictionCol(self, value): return self._set(predictionCol=value)
    def getPredictionCol(self): return self.getOrDefault(self.predictionCol)
    def setScoreCol(self, value): return self._set(scoreCol=value)
    def getScoreCol(self): return self.getOrDefault(self.scoreCol)


class _ExtendedIsolationForestParams(_IsolationForestParams):
    extensionLevel = Param(
        Params._dummy(), "extensionLevel",
        "Extension level of the random hyperplane: extensionLevel + 1 coordinates are non-zero. "
        "0 = axis-aligned splits; numFeatures - 1 (of the per-tree subspace) = fully extended. "
        "If not set, defaults to fully extended at fit time.", typeConverter=TypeConverters.toInt)

    def setExtensionLevel(self, value): return self._set(extensionLevel=value)
    def getExtensionLevel(self): return self.getOrDefault(self.extensionLevel)
