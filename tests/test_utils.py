"""Ports of UtilsTest plus the parameter-resolution rules of SharedTrainLogic."""
import math

import numpy as np
import pytest

from pyspark_isolation_forest.utils import (avg_path_length, height_limit, resolve_params,
                                            sample_fraction, scores_from_path_lengths)


def test_avg_path_length_matches_scala_utils_test_exactly():
    assert avg_path_length(0) == np.float32(0.0)
    assert avg_path_length(1) == np.float32(0.0)
    assert avg_path_length(2) == np.float32(0.15443134)
    assert avg_path_length(10) == np.float32(3.7488806)
    assert avg_path_length(2 ** 63 - 1) == np.float32(86.49098)       # Long.MaxValue


def test_avg_path_length_is_float32_and_vectorised():
    out = avg_path_length(np.array([0, 1, 2, 10]))
    assert out.dtype == np.float32
    assert out[2] == np.float32(0.15443134)
    assert isinstance(avg_path_length(5), np.float32)


@pytest.mark.parametrize("n,expected", [(1, 0), (2, 1), (3, 2), (4, 2), (5, 3), (256, 8), (257, 9), (1000, 10)])
def test_height_limit_is_ceil_log2(n, expected):
    assert height_limit(n) == expected


def test_height_limit_equals_exact_ceil_log2_for_realistic_sizes():
    for n in range(2, 20000):
        assert height_limit(n) == (n - 1).bit_length()


class TestResolveParams:
    def test_fraction_and_count_semantics(self):
        r = resolve_params(0.5, 0.25, total_num_features=7, total_num_samples=1000)
        assert (r.num_features, r.num_samples) == (3, 250)           # floor(0.5*7)=3
        r = resolve_params(4.0, 300.0, 7, 1000)
        assert (r.num_features, r.num_samples) == (4, 300)           # > 1.0 is a count
        r = resolve_params(2.9, 300.9, 7, 1000)
        assert (r.num_features, r.num_samples) == (2, 300)           # floor

    def test_exactly_one_is_a_fraction(self):
        r = resolve_params(1.0, 1.0, 5, 123)
        assert (r.num_features, r.num_samples) == (5, 123)

    @pytest.mark.parametrize("max_features", [0.05, 8.0])
    def test_invalid_max_features(self, max_features):
        with pytest.raises(ValueError, match="maxFeatures"):
            resolve_params(max_features, 10, 5, 100)

    @pytest.mark.parametrize("max_samples", [1.5, 0.001, 101.0])
    def test_invalid_max_samples(self, max_samples):
        with pytest.raises(ValueError, match="maxSamples"):
            resolve_params(1.0, max_samples, 5, 100)

    def test_max_samples_larger_than_dataset_is_an_error_not_clamped(self):
        with pytest.raises(ValueError, match="only 100 samples"):
            resolve_params(1.0, 256, 5, 100)


def test_sample_fraction_oversamples_seven_sigma_and_caps_at_one():
    assert sample_fraction(256, 10_000) == pytest.approx((256 + 7 * math.sqrt(256)) / 10_000)
    assert sample_fraction(256, 300) == 1.0
    assert sample_fraction(2, 2) == 1.0


def test_score_formula():
    # path == c(numSamples)  ->  score = 2**-1 = 0.5 ;  path == 0 -> score = 1
    c = float(avg_path_length(256))
    s = scores_from_path_lengths(np.array([c, 0.0], dtype=np.float32), 1, 256)
    assert s[0] == pytest.approx(0.5, abs=1e-6) and s[1] == 1.0
