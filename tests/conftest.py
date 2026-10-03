import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# Python workers must be able to import the package.
os.environ["PYTHONPATH"] = ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")


@pytest.fixture(scope="session")
def spark():
    from pyspark.sql import SparkSession

    session = (SparkSession.builder.master("local[2]").appName("pyspark-isolation-forest-tests")
               .config("spark.sql.shuffle.partitions", "4")
               .config("spark.default.parallelism", "4")
               .config("spark.ui.enabled", "false")
               .config("spark.driver.memory", "2g")
               .config("spark.sql.execution.arrow.maxRecordsPerBatch", "5000")
               .getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    java = session._jvm.System.getProperty("java.version")
    major = int(java.split(".")[0]) if not java.startswith("1.") else int(java.split(".")[1])
    if major >= 21 and session.version.startswith("3."):
        session.stop()
        pytest.exit(f"Spark {session.version} bundles Arrow 12, which does not work on JDK {java}. "
                    "Run the tests with JDK 17 (set JAVA_HOME).", returncode=2)
    yield session
    session.stop()


@pytest.fixture(scope="session")
def mammography(spark):
    return load_labeled_csv(spark, os.path.join(RESOURCES, "mammography.csv"))


@pytest.fixture(scope="session")
def shuttle(spark):
    return load_labeled_csv(spark, os.path.join(RESOURCES, "shuttle.csv"))


def read_csv_matrix(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append([float(x) for x in line.split(",")])
    return np.asarray(rows)


def load_labeled_csv(spark, path, partitions=4):
    from pyspark.ml.linalg import Vectors

    m = read_csv_matrix(path)
    rows = [(Vectors.dense(r[:-1].tolist()), float(r[-1])) for r in m]
    return spark.createDataFrame(rows, ["features", "label"]).repartition(partitions).cache()


def make_df(spark, X, y=None, partitions=4):
    from pyspark.ml.linalg import Vectors

    X = np.asarray(X, dtype=float)
    if y is None:
        rows = [(Vectors.dense(r.tolist()),) for r in X]
        return spark.createDataFrame(rows, ["features"]).repartition(partitions)
    rows = [(Vectors.dense(r.tolist()), float(l)) for r, l in zip(X, y)]
    return spark.createDataFrame(rows, ["features", "label"]).repartition(partitions)


def auroc(scores, labels):
    """Rank-based (Mann-Whitney) area under the ROC curve, ties handled by average ranks."""
    from scipy.stats import rankdata  # noqa: F401
    scores, labels = np.asarray(scores), np.asarray(labels)
    ranks = rankdata(scores)
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
