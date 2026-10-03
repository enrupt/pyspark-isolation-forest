"""Statistical tests of the bagging stage (ports of BaggedPointTest: distribution checks, not bit-equality)."""
import numpy as np
import pytest

from conftest import make_df
from pyspark_isolation_forest import training
from pyspark_isolation_forest.training import _BAG_SCHEMA, _make_bagging_fn


def bag(spark, n, d, num_trees, rate, bootstrap, seed, partitions=4):
    X = np.arange(n * d, dtype=float).reshape(n, d)
    df = make_df(spark, X, partitions=partitions)
    out = training.features_as_array(df, "features").mapInArrow(
        _make_bagging_fn(num_trees, rate, bootstrap, seed), _BAG_SCHEMA)
    return out.toPandas()


def test_without_replacement_counts_follow_binomial(spark):
    n, trees, rate = 20000, 12, 0.5
    pdf = bag(spark, n, 2, trees, rate, False, 1)
    counts = pdf.groupby("treeId").size()
    assert len(counts) == trees
    sd = np.sqrt(n * rate * (1 - rate))
    assert (abs(counts - n * rate) < 5 * sd).all()
    # no row appears twice in the same tree (Bernoulli, not Poisson)
    assert not pdf.duplicated(["treeId", "part", "seq"]).any()


def test_with_replacement_counts_follow_poisson_and_repeat_rows(spark):
    n, trees, rate = 20000, 12, 0.5
    pdf = bag(spark, n, 2, trees, rate, True, 1)
    counts = pdf.groupby("treeId").size()
    assert (abs(counts - n * rate) < 5 * np.sqrt(n * rate)).all()
    assert pdf.duplicated(["treeId", "part", "seq"]).any()
    per_row = pdf.groupby(["treeId", "part", "seq"]).size()
    assert per_row.max() >= 2


def test_rate_one_without_replacement_keeps_every_row_in_every_tree(spark):
    pdf = bag(spark, 500, 2, 3, 1.0, False, 1)
    assert pdf.groupby("treeId").size().tolist() == [500, 500, 500]


def test_single_tree_rate_one_is_the_no_sampling_special_case(spark):
    pdf = bag(spark, 300, 2, 1, 1.0, False, 1)
    assert len(pdf) == 300 and set(pdf.treeId) == {0}


def test_small_rate_still_produces_valid_output(spark):
    pdf = bag(spark, 5000, 2, 4, 0.002, False, 1)
    assert len(pdf) > 0 and set(pdf.treeId) <= {0, 1, 2, 3}


def test_trees_get_independent_samples(spark):
    pdf = bag(spark, 20000, 1, 6, 0.5, False, 1)
    sets = [set(zip(g.part, g.seq)) for _, g in pdf.groupby("treeId")]
    overlap = len(sets[0] & sets[1]) / len(sets[0])
    assert overlap == pytest.approx(0.5, abs=0.03)             # independent Bernoulli(0.5) draws


def test_partitions_use_different_random_streams(spark):
    # Same data in every partition -> if streams were shared the keep-masks would be identical.
    pdf = bag(spark, 8000, 1, 1, 0.5, False, 1, partitions=2)
    keep = {p: set(g.seq) for p, g in pdf.groupby("part")}
    assert len(keep) == 2
    a, b = sorted(keep)
    assert keep[a] != keep[b]


def test_same_seed_is_deterministic_and_seed_changes_sample(spark):
    a, b = bag(spark, 3000, 2, 5, 0.3, False, 11), bag(spark, 3000, 2, 5, 0.3, False, 11)
    c = bag(spark, 3000, 2, 5, 0.3, False, 12)
    key = lambda p: sorted(zip(p.treeId, p.part, p.seq))
    assert key(a) == key(b)
    assert key(a) != key(c)


def test_features_are_carried_through_unchanged(spark):
    pdf = bag(spark, 1000, 3, 2, 0.5, False, 1)
    # row r (global order is lost after repartition) -> feature triple is (3k, 3k+1, 3k+2)
    f = np.stack(pdf.f.to_numpy())
    assert np.all(f[:, 1] == f[:, 0] + 1) and np.all(f[:, 2] == f[:, 0] + 2)
