"""Standard isolation tree: unit tests without Spark (ports of IsolationTreeTest + edge cases)."""
import numpy as np
import pytest

from pyspark_isolation_forest.tree import IsolationTree, Leaf, Split, generate_isolation_tree
from pyspark_isolation_forest.utils import avg_path_length, height_limit


def gauss(n, d, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)


def used_features(t):
    return set(t.feature[t.feature >= 0].tolist())


def test_leaf_and_split_validation():
    with pytest.raises(ValueError):
        Leaf(0)
    with pytest.raises(ValueError):
        Split(Leaf(1), Leaf(1), -1, 0.5)
    assert Split(Leaf(1), Split(Leaf(1), Leaf(1), 0, 0.1), 0, 0.5).depth == 2


def test_all_training_points_end_up_in_exactly_one_leaf():
    X = gauss(256, 4)
    t = IsolationTree.fit(X, 7, np.arange(4))
    assert t.num_instances[t.feature < 0].sum() == 256


def test_depth_never_exceeds_height_limit():
    for n in (2, 3, 17, 100, 256, 300):
        t = IsolationTree.fit(gauss(n, 3, seed=n), 1, np.arange(3))
        assert t.depth <= height_limit(n)


def test_tree_is_fully_determined_by_seed():
    X = gauss(128, 3)
    a, b = IsolationTree.fit(X, 5, np.arange(3)), IsolationTree.fit(X, 5, np.arange(3))
    assert a.to_row() == b.to_row()
    c = IsolationTree.fit(X, 6, np.arange(3))
    assert a.to_row() != c.to_row()


def test_splits_use_only_the_given_feature_subspace():
    X = gauss(256, 6)
    t = IsolationTree.fit(X, 3, np.array([1, 4]))
    assert used_features(t) <= {1, 4} and used_features(t)


def test_split_values_lie_strictly_inside_the_data_range_of_the_feature():
    X = gauss(256, 3)
    t = IsolationTree.fit(X, 3, np.arange(3))
    root = t.feature[0]
    assert X[:, root].min() <= t.threshold[0] <= X[:, root].max()


def test_split_semantics_left_is_strictly_less_than_threshold():
    # Two distinct values {0, 1}: any split in (0, 1] puts the 0s left and the 1s right.
    X = np.array([[0.0]] * 5 + [[1.0]] * 5, dtype=np.float32)
    t = IsolationTree.fit(X, 1, np.array([0]))
    assert t.feature[0] == 0 and 0.0 < t.threshold[0] < 1.0
    assert t.num_instances[t.left[0]] == 5 and t.num_instances[t.right[0]] == 5


def test_constant_feature_is_skipped_and_never_split_on():
    X = gauss(256, 3)
    X[:, 1] = 7.0
    for seed in range(5):
        t = IsolationTree.fit(X, seed, np.arange(3))
        assert 1 not in used_features(t)


def test_all_features_constant_gives_single_leaf():
    X = np.full((50, 3), 2.5, dtype=np.float32)
    t = IsolationTree.fit(X, 1, np.arange(3))
    assert t.num_nodes == 1 and t.num_instances[0] == 50 and t.depth == 0


def test_duplicate_points_collapse_into_one_leaf():
    X = np.vstack([np.zeros((20, 2)), np.ones((20, 2))]).astype(np.float32)
    t = IsolationTree.fit(X, 1, np.arange(2))
    leaves = t.num_instances[t.feature < 0]
    assert sorted(leaves.tolist()) == [20, 20]       # two distinct points -> one split -> two leaves


@pytest.mark.parametrize("n", [1, 2])
def test_very_small_datasets(n):
    X = gauss(n, 2)
    t = IsolationTree.fit(X, 1, np.arange(2))
    assert t.num_instances[t.feature < 0].sum() == n
    if n == 1:
        assert t.num_nodes == 1


def test_path_length_of_a_hand_built_tree_matches_expectation():
    # root: x0 < 0.5 -> left leaf(3) ; right: x1 < 1.0 -> leaf(2) / leaf(1)
    root = Split(Leaf(3), Split(Leaf(2), Leaf(1), 1, 1.0), 0, 0.5)
    t = IsolationTree.from_node(root)
    X = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    got = t.path_length(X)
    exp = np.array([1 + avg_path_length(3), 2 + avg_path_length(2), 2 + avg_path_length(1)], dtype=np.float32)
    assert got.dtype == np.float32 and np.array_equal(got, exp)


def test_point_exactly_on_threshold_goes_right():
    t = IsolationTree.from_node(Split(Leaf(5), Leaf(7), 0, 0.5))
    got = t.path_length(np.array([[0.5], [0.4999]], dtype=np.float32))
    assert got[0] == 1 + avg_path_length(7) and got[1] == 1 + avg_path_length(5)


def test_features_are_compared_as_float32_widened_to_float64():
    thr = 0.1                                     # not representable in float32
    t = IsolationTree.from_node(Split(Leaf(5), Leaf(7), 0, thr))
    x = np.array([[0.1]], dtype=np.float32)       # float32(0.1) = 0.10000000149 > 0.1 -> right
    assert t.path_length(x)[0] == 1 + avg_path_length(7)


def test_outliers_are_isolated_faster_than_inliers():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(size=(255, 2)), [[30.0, 30.0]]]).astype(np.float32)
    lens = np.mean([IsolationTree.fit(X, s, np.arange(2)).path_length(
        np.array([[0, 0], [30, 30]], dtype=np.float32)) for s in range(30)], axis=0)
    assert lens[1] < lens[0]


def test_flatten_roundtrip_and_node_roundtrip():
    t = IsolationTree.fit(gauss(200, 3), 9, np.arange(3))
    t2 = IsolationTree.from_row(t.to_row())
    assert t2.to_row() == t.to_row()
    t3 = IsolationTree.from_node(t.to_node())
    assert t3.to_row() == t.to_row()
    X = gauss(50, 3, seed=5)
    assert np.array_equal(t.path_length(X), t3.path_length(X))


def test_invalid_arrays_rejected():
    with pytest.raises(ValueError):
        IsolationTree([0], [0.0, 1.0], [-1], [-1], [1])
    with pytest.raises(ValueError):
        IsolationTree([], [], [], [], [])
