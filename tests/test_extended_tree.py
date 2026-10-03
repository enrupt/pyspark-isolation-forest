"""Extended isolation tree unit tests (ports of ExtendedIsolationTreeTest)."""
import numpy as np
import pytest

from pyspark_isolation_forest.extended_tree import (ExtLeaf, ExtSplit, ExtendedIsolationTree,
                                                    SplitHyperplane, generate_extended_isolation_tree)
from pyspark_isolation_forest.utils import avg_path_length, height_limit


def gauss(n, d, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)


def internal(t):
    return np.nonzero(t.left >= 0)[0]


def test_split_hyperplane_validation():
    with pytest.raises(ValueError, match="non-empty"):
        SplitHyperplane([], [], 0.0)
    with pytest.raises(ValueError, match="same length"):
        SplitHyperplane([0, 1], [1.0], 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        SplitHyperplane([-1], [1.0], 0.0)
    with pytest.raises(ValueError, match="distinct"):
        SplitHyperplane([1, 1], [1.0, 1.0], 0.0)
    with pytest.raises(ValueError, match="sorted"):
        SplitHyperplane([2, 1], [1.0, 1.0], 0.0)
    with pytest.raises(ValueError):
        ExtLeaf(-1)
    assert ExtLeaf(0).num_instances == 0                      # zero-size leaves are legal


def test_split_hyperplane_dot_matches_dense_reference():
    X = gauss(20, 6, seed=3)
    hp = SplitHyperplane([1, 4], [0.6, -0.8], 0.0)
    dense = np.zeros(6, dtype=np.float32)
    dense[[1, 4]] = np.float32([0.6, -0.8])
    ref = (X * dense).astype(np.float64)[:, [1, 4]].sum(axis=1)
    assert np.allclose(hp.dot(X), ref, atol=1e-12)


def test_dot_multiplies_in_float32_and_accumulates_in_float64():
    hp = SplitHyperplane([0, 1], [np.float32(0.1), np.float32(0.2)], 0.0)
    X = np.array([[np.float32(0.3), np.float32(0.7)]], dtype=np.float32)
    expected = float(np.float32(0.1) * np.float32(0.3)) + float(np.float32(0.2) * np.float32(0.7))
    assert hp.dot(X)[0] == expected


@pytest.mark.parametrize("level,k", [(0, 1), (1, 2), (3, 4), (5, 6)])
def test_extension_level_controls_number_of_nonzero_coordinates(level, k):
    t = ExtendedIsolationTree.fit(gauss(200, 6), 1, np.arange(6), level)
    assert t.k == k
    for i in internal(t):
        assert np.count_nonzero(t.weights[i]) == k
        assert len(set(t.indices[i].tolist())) == k


def test_extension_level_larger_than_dimension_is_capped():
    assert ExtendedIsolationTree.fit(gauss(100, 3), 1, np.arange(3), 10).k == 3


def test_hyperplane_normals_are_l2_normalised_and_indices_sorted():
    t = ExtendedIsolationTree.fit(gauss(256, 5), 2, np.arange(5), 4)
    for i in internal(t):
        assert abs(np.linalg.norm(t.weights[i].astype(np.float64)) - 1.0) < 1e-6
        assert np.all(np.diff(t.indices[i]) > 0)
        hp = t.hyperplane(i)
        assert isinstance(hp, SplitHyperplane)


def test_hyperplanes_stay_inside_the_feature_subspace():
    t = ExtendedIsolationTree.fit(gauss(256, 8), 2, np.array([1, 3, 6]), 2)
    for i in internal(t):
        assert set(t.indices[i].tolist()) <= {1, 3, 6}


def test_extension_level_zero_is_axis_aligned():
    t = ExtendedIsolationTree.fit(gauss(256, 4), 1, np.arange(4), 0)
    for i in internal(t):
        assert t.indices[i].shape == (1,) and abs(abs(float(t.weights[i][0])) - 1.0) < 1e-6


def test_zero_size_leaf_has_zero_path_contribution():
    root = ExtSplit(ExtLeaf(0), ExtLeaf(4), SplitHyperplane([0], [1.0], 0.0))
    t = ExtendedIsolationTree.from_node(root, 1)
    X = np.array([[-1.0], [1.0]], dtype=np.float32)
    got = t.path_length(X)
    assert got[0] == np.float32(1.0)                          # 1 + c(0) = 1
    assert got[1] == np.float32(1.0) + avg_path_length(4)


def test_degenerate_splits_do_not_retry_and_may_create_empty_children():
    # Constant data: every hyperplane puts all points on one side -> empty sibling leaves exist.
    X = np.full((32, 2), 3.0, dtype=np.float32)
    t = ExtendedIsolationTree.fit(X, 1, np.arange(2), 1)
    leaves = t.left < 0
    assert (t.num_instances[leaves] == 0).any()
    assert t.num_instances[leaves].sum() == 32


def test_depth_limit_and_leaf_totals():
    X = gauss(256, 4)
    t = ExtendedIsolationTree.fit(X, 1, np.arange(4), 3)
    assert t.depth <= height_limit(256)
    assert t.num_instances[t.left < 0].sum() == 256


def test_determinism_by_seed():
    X = gauss(128, 3)
    a = ExtendedIsolationTree.fit(X, 4, np.arange(3), 2)
    assert a.to_row() == ExtendedIsolationTree.fit(X, 4, np.arange(3), 2).to_row()
    assert a.to_row() != ExtendedIsolationTree.fit(X, 5, np.arange(3), 2).to_row()


def test_roundtrip_and_vectorised_path_matches_node_traversal():
    X = gauss(200, 4)
    t = ExtendedIsolationTree.fit(X, 9, np.arange(4), 3)
    t2 = ExtendedIsolationTree.from_row(t.to_row())
    Q = gauss(40, 4, seed=11)
    assert np.array_equal(t.path_length(Q), t2.path_length(Q))

    # independent scalar traversal over the flat arrays
    def scalar(x):
        node, steps = 0, np.float32(0.0)
        while t.left[node] >= 0:
            dp = sum(float(np.float32(t.weights[node][j]) * np.float32(x[t.indices[node][j]])) for j in range(t.k))
            node = t.left[node] if dp < t.offset[node] else t.right[node]
            steps += np.float32(1.0)
        return steps + avg_path_length(int(t.num_instances[node]))

    assert np.array_equal(t.path_length(Q), np.array([scalar(q) for q in Q], dtype=np.float32))


def test_off_diagonal_points_are_isolated_faster_with_full_extension_on_correlated_data():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(256, 1))
    X = np.hstack([z, z + 0.1 * rng.normal(size=(256, 1))]).astype(np.float32)
    on, off = np.array([[2.0, 2.0]], dtype=np.float32), np.array([[2.0, -2.0]], dtype=np.float32)
    on_l, off_l = [], []
    for s in range(60):
        t = ExtendedIsolationTree.fit(X, s, np.arange(2), 1)
        on_l.append(t.path_length(on)[0]); off_l.append(t.path_length(off)[0])
    assert np.mean(off_l) < np.mean(on_l)
