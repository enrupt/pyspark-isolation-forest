"""Extended isolation tree with random sparse hyperplane splits
(port of ExtendedIsolationTree.scala / ExtendedNodes.scala / ExtendedUtils.scala).

Differences from the standard tree that are deliberately preserved:
  * no retry on constant features (a split is always attempted),
  * an empty child is allowed and becomes a leaf with numInstances = 0 (c(0) = 0),
  * hyperplane normals are L2-normalised, stored as float32; the dot product multiplies in
    float32 and accumulates in float64; the split offset is a float64.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .utils import avg_path_length, height_limit

NULL_NODE = -1


@dataclass
class SplitHyperplane:
    """Sparse hyperplane: `indices` (sorted, distinct, in the original feature space), float32 `weights`, float64 `offset`."""
    indices: np.ndarray
    weights: np.ndarray
    offset: float

    def __post_init__(self):
        self.indices = np.asarray(self.indices, dtype=np.int64)
        self.weights = np.asarray(self.weights, dtype=np.float32)
        if self.indices.size == 0:
            raise ValueError("indices must be non-empty.")
        if self.indices.shape != self.weights.shape:
            raise ValueError("indices and weights must have the same length.")
        if (self.indices < 0).any():
            raise ValueError("indices must be non-negative.")
        if np.unique(self.indices).size != self.indices.size:
            raise ValueError("indices must be distinct.")
        if not np.all(self.indices[:-1] < self.indices[1:]):
            raise ValueError("indices must be sorted in ascending order.")

    def dot(self, X: np.ndarray) -> np.ndarray:
        """float32 products, float64 accumulation. X is (n, d) float32 -> (n,) float64."""
        sub = np.asarray(X, dtype=np.float32)[:, self.indices]
        return (sub * self.weights).astype(np.float64).sum(axis=1)


@dataclass
class ExtLeaf:
    num_instances: int

    def __post_init__(self):
        if self.num_instances < 0:
            raise ValueError(f"parameter numInstances must be >= 0, but given invalid value {self.num_instances}")

    @property
    def depth(self) -> int:
        return 0


@dataclass
class ExtSplit:
    left: "ExtNode"
    right: "ExtNode"
    hyperplane: SplitHyperplane

    @property
    def depth(self) -> int:
        return 1 + max(self.left.depth, self.right.depth)


ExtNode = "ExtLeaf | ExtSplit"


def generate_extended_isolation_tree(data: np.ndarray, height_lim: int, rng: np.random.Generator,
                                     feature_indices: np.ndarray, extension_level: int):
    feature_indices = np.asarray(feature_indices, dtype=np.int64)
    dim = feature_indices.shape[0]
    n_non_zero = min(extension_level + 1, dim)

    def build(rows: np.ndarray, depth: int):
        n = rows.shape[0]
        if depth >= height_lim or n <= 1:
            return ExtLeaf(n)
        chosen_in_subspace = rng.permutation(dim)[:n_non_zero]
        sparse_indices = feature_indices[chosen_in_subspace]
        raw = rng.standard_normal(n_non_zero)
        norm = float(np.sqrt((raw * raw).sum()))
        if norm == 0.0:
            return ExtLeaf(n)
        weights = (raw / norm).astype(np.float32)
        sub = rows[:, sparse_indices].astype(np.float64)
        mn, mx = sub.min(axis=0), sub.max(axis=0)
        u = rng.random(n_non_zero)
        intercept = np.where(mn == mx, mn, mn + u * (mx - mn))
        offset = float((weights.astype(np.float64) * intercept).sum())
        order = np.argsort(sparse_indices, kind="stable")  # canonical storage order
        hp = SplitHyperplane(sparse_indices[order], weights[order], offset)
        go_left = hp.dot(rows) < offset
        left_child = build(rows[go_left], depth + 1)
        right_child = build(rows[~go_left], depth + 1)
        return ExtSplit(left_child, right_child, hp)

    return build(np.ascontiguousarray(data, dtype=np.float32), 0)


class ExtendedIsolationTree:
    """Pre-order flattened extended tree. Every internal node stores exactly `k` (index, weight)
    pairs (k = min(extensionLevel + 1, subspace dim)); leaves are zero-padded and have left == -1."""

    __slots__ = ("indices", "weights", "offset", "left", "right", "num_instances", "k", "_leaf_c")

    def __init__(self, indices, weights, offset, left, right, num_instances, k):
        self.k = int(k)
        self.left = np.asarray(left, dtype=np.int32)
        n = len(self.left)
        if n == 0 or self.k <= 0:
            raise ValueError("inconsistent extended tree arrays")
        self.indices = np.asarray(indices, dtype=np.int32).reshape(n, self.k)
        self.weights = np.asarray(weights, dtype=np.float32).reshape(n, self.k)
        self.offset = np.asarray(offset, dtype=np.float64)
        self.right = np.asarray(right, dtype=np.int32)
        self.num_instances = np.asarray(num_instances, dtype=np.int64)
        if not (len(self.offset) == len(self.right) == len(self.num_instances) == n):
            raise ValueError("inconsistent extended tree arrays")
        leaf_c = np.zeros(n, dtype=np.float32)
        leaves = self.left < 0
        leaf_c[leaves] = avg_path_length(self.num_instances[leaves])
        self._leaf_c = leaf_c

    @classmethod
    def from_node(cls, root, k: int) -> "ExtendedIsolationTree":
        idx_rows, w_rows, offset, left, right, num = [], [], [], [], [], []

        def visit(node) -> int:
            i = len(left)
            idx_rows.append(np.zeros(k, np.int32)); w_rows.append(np.zeros(k, np.float32))
            offset.append(0.0); left.append(NULL_NODE); right.append(NULL_NODE); num.append(-1)
            if isinstance(node, ExtLeaf):
                num[i] = node.num_instances
            else:
                hp = node.hyperplane
                idx_rows[i][: hp.indices.size] = hp.indices
                w_rows[i][: hp.weights.size] = hp.weights
                offset[i] = hp.offset
                left[i] = visit(node.left)
                right[i] = visit(node.right)
            return i

        visit(root)
        return cls(np.array(idx_rows), np.array(w_rows), offset, left, right, num, k)

    @classmethod
    def fit(cls, data, seed, feature_indices, extension_level: int,
            rng: Optional[np.random.Generator] = None) -> "ExtendedIsolationTree":
        data = np.asarray(data, dtype=np.float32)
        rng = rng if rng is not None else np.random.default_rng(seed)
        root = generate_extended_isolation_tree(data, height_limit(data.shape[0]), rng,
                                                feature_indices, extension_level)
        k = min(extension_level + 1, len(feature_indices))
        return cls.from_node(root, k)

    # ---- inspection --------------------------------------------------------------------
    @property
    def num_nodes(self) -> int:
        return len(self.left)

    @property
    def depth(self) -> int:
        depth = np.zeros(self.num_nodes, dtype=np.int64)
        for i in range(self.num_nodes):
            if self.left[i] >= 0:
                depth[self.left[i]] = depth[i] + 1
                depth[self.right[i]] = depth[i] + 1
        return int(depth.max())

    def hyperplane(self, node: int) -> SplitHyperplane:
        return SplitHyperplane(self.indices[node], self.weights[node], float(self.offset[node]))

    # ---- scoring -----------------------------------------------------------------------
    def path_length(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]
        node = np.zeros(n, dtype=np.int64)
        steps = np.zeros(n, dtype=np.float32)
        while True:
            active = np.nonzero(self.left[node] >= 0)[0]
            if active.size == 0:
                break
            nd = node[active]
            gathered = X[active[:, None], self.indices[nd]]          # (m, k) float32
            dp = (gathered * self.weights[nd]).astype(np.float64).sum(axis=1)
            go_left = dp < self.offset[nd]
            node[active] = np.where(go_left, self.left[nd], self.right[nd])
            steps[active] += np.float32(1.0)
        return steps + self._leaf_c[node]

    # ---- (de)serialisation -------------------------------------------------------------
    def to_row(self) -> dict:
        return {"k": self.k, "indices": self.indices.reshape(-1).tolist(),
                "weights": self.weights.reshape(-1).tolist(), "offset": self.offset.tolist(),
                "left": self.left.tolist(), "right": self.right.tolist(),
                "numInstances": self.num_instances.tolist()}

    @classmethod
    def from_row(cls, row) -> "ExtendedIsolationTree":
        return cls(row["indices"], row["weights"], row["offset"], row["left"], row["right"],
                   row["numInstances"], row["k"])
