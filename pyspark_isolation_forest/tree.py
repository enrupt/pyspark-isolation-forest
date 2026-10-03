"""Standard isolation tree (port of IsolationTree.scala / Nodes.scala).

Training builds a tree of `Leaf` / `Split` nodes, which is then flattened into parallel numpy
arrays (`IsolationTree`) for compact serialisation and vectorised scoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .utils import avg_path_length, height_limit

NULL_NODE = -1


# --------------------------------------------------------------------------------------------
# Build-time node representation
# --------------------------------------------------------------------------------------------
@dataclass
class Leaf:
    num_instances: int

    def __post_init__(self):
        if self.num_instances <= 0:
            raise ValueError(f"parameter numInstances must be >0, but given invalid value {self.num_instances}")

    @property
    def depth(self) -> int:
        return 0


@dataclass
class Split:
    left: "Node"
    right: "Node"
    split_attribute: int
    split_value: float

    def __post_init__(self):
        if self.split_attribute < 0:
            raise ValueError(f"parameter splitAttribute must be >=0, but given invalid value {self.split_attribute}")

    @property
    def depth(self) -> int:
        return 1 + max(self.left.depth, self.right.depth)


Node = "Leaf | Split"


def generate_isolation_tree(data: np.ndarray, height_lim: int, rng: np.random.Generator,
                            feature_indices: np.ndarray):
    """Recursive build, following IsolationTree.generateIsolationTree.

    `data` is an (n, d) float32 array. At every node, candidate features are tried in a random
    order and the first non-constant one is used (constant features are skipped and retried).
    The node is a leaf if no feature can be split, the height limit is reached, or n <= 1.
    """
    feature_indices = np.asarray(feature_indices, dtype=np.int64)

    def build(rows: np.ndarray, depth: int):
        n = rows.shape[0]
        if depth >= height_lim or n <= 1:
            return Leaf(n)
        chosen, split_value = -1, 0.0
        for f in rng.permutation(feature_indices):
            col = rows[:, f]
            mn, mx = float(col.min()), float(col.max())
            if mn != mx:
                chosen = int(f)
                split_value = (mx - mn) * float(rng.random()) + mn
                break
        if chosen == -1:
            return Leaf(n)
        col64 = rows[:, chosen].astype(np.float64)
        go_left = col64 < split_value
        left, right = rows[go_left], rows[~go_left]
        if left.shape[0] == 0 or right.shape[0] == 0:
            # Only reachable if the uniform draw is exactly 0.0 (split == min); upstream would fail.
            return Leaf(n)
        return Split(build(left, depth + 1), build(right, depth + 1), chosen, split_value)

    return build(np.ascontiguousarray(data, dtype=np.float32), 0)


# --------------------------------------------------------------------------------------------
# Flat, serialisable tree
# --------------------------------------------------------------------------------------------
class IsolationTree:
    """Pre-order flattened tree. Leaves have feature == -1 and children == -1."""

    __slots__ = ("feature", "threshold", "left", "right", "num_instances", "_leaf_c")

    def __init__(self, feature, threshold, left, right, num_instances):
        self.feature = np.asarray(feature, dtype=np.int32)
        self.threshold = np.asarray(threshold, dtype=np.float64)
        self.left = np.asarray(left, dtype=np.int32)
        self.right = np.asarray(right, dtype=np.int32)
        self.num_instances = np.asarray(num_instances, dtype=np.int64)
        n = len(self.feature)
        if not (len(self.threshold) == len(self.left) == len(self.right) == len(self.num_instances) == n) or n == 0:
            raise ValueError("inconsistent tree arrays")
        leaf_c = np.zeros(n, dtype=np.float32)
        leaves = self.feature < 0
        leaf_c[leaves] = avg_path_length(self.num_instances[leaves])
        self._leaf_c = leaf_c

    # ---- construction ------------------------------------------------------------------
    @classmethod
    def from_node(cls, root) -> "IsolationTree":
        feature, threshold, left, right, num = [], [], [], [], []

        def visit(node) -> int:
            idx = len(feature)
            feature.append(-1); threshold.append(0.0); left.append(NULL_NODE)
            right.append(NULL_NODE); num.append(-1)
            if isinstance(node, Leaf):
                num[idx] = node.num_instances
            else:
                feature[idx] = node.split_attribute
                threshold[idx] = node.split_value
                left[idx] = visit(node.left)
                right[idx] = visit(node.right)
            return idx

        visit(root)
        return cls(feature, threshold, left, right, num)

    @classmethod
    def fit(cls, data: np.ndarray, seed, feature_indices, rng: Optional[np.random.Generator] = None):
        """Fit one tree on `data` (n, d). `seed` may be an int or a list of ints (SeedSequence entropy)."""
        data = np.asarray(data, dtype=np.float32)
        rng = rng if rng is not None else np.random.default_rng(seed)
        root = generate_isolation_tree(data, height_limit(data.shape[0]), rng, feature_indices)
        return cls.from_node(root)

    # ---- inspection --------------------------------------------------------------------
    @property
    def num_nodes(self) -> int:
        return len(self.feature)

    @property
    def depth(self) -> int:
        depth = np.zeros(self.num_nodes, dtype=np.int64)
        for i in range(self.num_nodes):  # pre-order: parents precede children
            if self.feature[i] >= 0:
                depth[self.left[i]] = depth[i] + 1
                depth[self.right[i]] = depth[i] + 1
        return int(depth.max())

    def to_node(self):
        def build(i):
            if self.feature[i] < 0:
                return Leaf(int(self.num_instances[i]))
            return Split(build(self.left[i]), build(self.right[i]), int(self.feature[i]), float(self.threshold[i]))
        return build(0)

    # ---- scoring -----------------------------------------------------------------------
    def path_length(self, X: np.ndarray) -> np.ndarray:
        """Vectorised path length for an (n, d) float32 batch: depth + c(leaf size). float32 result."""
        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]
        node = np.zeros(n, dtype=np.int64)
        steps = np.zeros(n, dtype=np.float32)
        while True:
            f = self.feature[node]
            active = np.nonzero(f >= 0)[0]
            if active.size == 0:
                break
            nd = node[active]
            go_left = X[active, f[active]].astype(np.float64) < self.threshold[nd]
            node[active] = np.where(go_left, self.left[nd], self.right[nd])
            steps[active] += np.float32(1.0)
        return steps + self._leaf_c[node]

    # ---- (de)serialisation -------------------------------------------------------------
    def to_row(self) -> dict:
        return {"feature": self.feature.tolist(), "threshold": self.threshold.tolist(),
                "left": self.left.tolist(), "right": self.right.tolist(),
                "numInstances": self.num_instances.tolist()}

    @classmethod
    def from_row(cls, row) -> "IsolationTree":
        return cls(row["feature"], row["threshold"], row["left"], row["right"], row["numInstances"])
