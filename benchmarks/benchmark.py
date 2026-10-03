"""Benchmark fit/transform time, peak memory and Spark job/stage counts.

Usage:  python benchmarks/benchmark.py --rows 10000 100000 1000000 --features 10 --trees 100 --out results.json
Data is generated inside Spark (no driver-side materialisation). Measured on whatever machine runs it:
in local mode the driver JVM also plays the executor, so memory is reported as JVM RSS and Python-worker RSS.
"""
import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONPATH"] = sys.path[0] + os.pathsep + os.environ.get("PYTHONPATH", "")

import psutil
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession, functions as F

from pyspark_isolation_forest import ExtendedIsolationForest, IsolationForest


class MemoryPoller(threading.Thread):
    def __init__(self, jvm_pid):
        super().__init__(daemon=True)
        self.jvm = psutil.Process(jvm_pid)
        self.stop_flag = threading.Event()
        self.peak_jvm = self.peak_workers = self.peak_driver_py = 0
        self.me = psutil.Process(os.getpid())

    def run(self):
        while not self.stop_flag.is_set():
            try:
                self.peak_jvm = max(self.peak_jvm, self.jvm.memory_info().rss)
                workers = sum(c.memory_info().rss for c in self.jvm.children(recursive=True))
                self.peak_workers = max(self.peak_workers, workers)
                self.peak_driver_py = max(self.peak_driver_py, self.me.memory_info().rss)
            except psutil.Error:
                pass
            time.sleep(0.1)


def measure(spark, label, fn):
    sc = spark.sparkContext
    group = f"bench-{label}-{time.time_ns()}"
    sc.setJobGroup(group, label)
    poller = MemoryPoller(sc._gateway.proc.pid)
    poller.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    poller.stop_flag.set(); poller.join()
    tr = sc.statusTracker()
    jobs = tr.getJobIdsForGroup(group)
    stages = sum(len(tr.getJobInfo(j).stageIds) for j in jobs if tr.getJobInfo(j))
    mb = lambda b: round(b / 2**20)
    return result, {"seconds": round(elapsed, 2), "jobs": len(jobs), "stages": stages,
                    "peak_jvm_rss_mb": mb(poller.peak_jvm), "peak_python_workers_rss_mb": mb(poller.peak_workers),
                    "peak_driver_python_rss_mb": mb(poller.peak_driver_py)}


def make_data(spark, n, d):
    cols = [F.randn(seed=i).alias(f"x{i}") for i in range(d)]
    df = spark.range(n).select(*cols)
    return VectorAssembler(inputCols=[f"x{i}" for i in range(d)], outputCol="features").transform(df) \
        .select("features").repartition(8).cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[10_000, 100_000])
    ap.add_argument("--features", type=int, default=10)
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--max-samples", type=float, default=256)
    ap.add_argument("--out", default="benchmark_results.json")
    a = ap.parse_args()
    spark = (SparkSession.builder.master("local[2]").appName("if-bench").config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", "8").config("spark.driver.memory", "2g")
             .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    results = []
    for n in a.rows:
        df = make_data(spark, n, a.features)
        df.count()                                           # materialise the cache outside the timings
        for name, cls, kw in [("IsolationForest", IsolationForest, {}),
                              ("ExtendedIsolationForest", ExtendedIsolationForest, {})]:
            est = cls(numEstimators=a.trees, maxSamples=a.max_samples, randomSeed=1, **kw)
            model, fit_m = measure(spark, f"fit-{name}-{n}", lambda: est.fit(df))
            _, tr_m = measure(spark, f"transform-{name}-{n}",
                              lambda: model.transform(df).agg(F.sum("outlierScore")).collect())
            est_c = cls(numEstimators=a.trees, maxSamples=a.max_samples, randomSeed=1, contamination=0.01,
                        contaminationError=0.0001, **kw)
            _, fitc_m = measure(spark, f"fitcont-{name}-{n}", lambda: est_c.fit(df))
            row = {"model": name, "rows": n, "features": a.features, "trees": a.trees,
                   "fit": fit_m, "transform": tr_m, "fit_with_contamination": fitc_m}
            results.append(row)
            print(json.dumps(row), flush=True)
        df.unpersist()
    json.dump(results, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
