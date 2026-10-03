"""Shared helpers: the c(n) normaliser, parameter resolution, schema validation.

Behaviour follows linkedin/isolation-forest (core/Utils.scala, core/SharedTrainLogic.scala).
All data and path-length arithmetic is float32, matching the Scala implementation.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

EULER_CONSTANT = np.float32(0.5772156649)


def avg_path_length(num_instances):
    """c(n): average path length of an unsuccessful BST search, in float32.

    Scalar in -> np.float32 out. Array in -> float32 array out.
    c(n) = 0 for n <= 1, else 2*(ln(n-1) + gamma) - 2*(n-1)/n.
    """
    scalar = np.ndim(num_instances) == 0
    n = np.atleast_1d(np.asarray(num_instances))
    nf = n.astype(np.float32)
    out = np.zeros(nf.shape, dtype=np.float32)
    mask = n > 1
    if mask.any():
        nm1 = nf[mask] - np.float32(1.0)
        log_term = np.log(nm1.astype(np.float64)).astype(np.float32)
        out[mask] = (
            np.float32(2.0) * (log_term + EULER_CONSTANT)
            - (np.float32(2.0) * nm1 / nf[mask])
        )
    return out[0] if scalar else out


def height_limit(n: int) -> int:
    """ceil(log2(n)) computed with the same formula as the Scala source."""
    if n <= 0:
        return 0
    return int(math.ceil(math.log10(float(n)) / math.log10(2.0)))


class ResolvedParams(NamedTuple):
    num_features: int
    total_num_features: int
    num_samples: int
    total_num_samples: int


def resolve_params(max_features: float, max_samples: float,
                   total_num_features: int, total_num_samples: int) -> ResolvedParams:
    """Resolve fractional/absolute maxFeatures and maxSamples (SharedTrainLogic.validateAndResolveParams)."""
    if max_features > 1.0:
        num_features = int(math.floor(max_features))
    else:
        num_features = int(math.floor(max_features * total_num_features))
    if num_features <= 0:
        raise ValueError(
            f"parameter maxFeatures given invalid value {max_features} specifying the use of "
            f"{num_features} features, but >0 features are required.")
    if num_features > total_num_features:
        raise ValueError(
            f"parameter maxFeatures given invalid value {max_features} specifying the use of "
            f"{num_features} features, but only {total_num_features} features are available.")

    if max_samples > 1.0:
        num_samples = int(math.floor(max_samples))
    else:
        num_samples = int(math.floor(max_samples * total_num_samples))
    if num_samples < 2:
        raise ValueError(
            f"parameter maxSamples given invalid value {max_samples} specifying the use of "
            f"{num_samples} samples, but >=2 samples are required.")
    if num_samples > total_num_samples:
        raise ValueError(
            f"parameter maxSamples given invalid value {max_samples} specifying the use of "
            f"{num_samples} samples, but only {total_num_samples} samples are in the input dataset.")
    return ResolvedParams(num_features, total_num_features, num_samples, total_num_samples)


def sample_fraction(num_samples: int, total_num_samples: int, n_sigma: float = 7.0) -> float:
    """Per-tree Bernoulli/Poisson rate: oversample by 7 sigma, capped at 1.0."""
    target = float(num_samples) + n_sigma * math.sqrt(float(num_samples))
    return min(target / float(total_num_samples), 1.0)


def validate_and_transform_schema(schema, features_col, prediction_col, score_col):
    """Mirror of Utils.validateAndTransformSchema. Returns the output StructType."""
    from pyspark.ml.linalg import VectorUDT
    from pyspark.sql.types import DoubleType, StructField, StructType

    names = schema.fieldNames()
    if features_col not in names:
        raise ValueError(f"Input column {features_col} does not exist.")
    if not isinstance(schema[features_col].dataType, VectorUDT):
        raise ValueError(f"Input column {features_col} is not of required type VectorUDT")
    if prediction_col in names:
        raise ValueError(f"Output column {prediction_col} already exists.")
    if score_col in names:
        raise ValueError(f"Output column {score_col} already exists.")
    if prediction_col == score_col:
        raise ValueError("predictionCol and scoreCol must be different columns.")
    return StructType(list(schema.fields)
                      + [StructField(prediction_col, DoubleType(), False),
                         StructField(score_col, DoubleType(), False)])


def scores_from_path_lengths(path_sum, num_trees: int, num_samples: int):
    """score = 2 ** (-mean_path / c(numSamples)); float32 until the final pow (as upstream)."""
    avg_path = avg_path_length(num_samples)
    mean_path = (path_sum.astype(np.float32) / np.float32(num_trees)).astype(np.float32)
    ratio = ((-mean_path) / avg_path).astype(np.float32)
    return np.power(2.0, ratio.astype(np.float64))
