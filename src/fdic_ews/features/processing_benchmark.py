"""Module 3 / Part 3 proof: Pandas vs Polars (eager, lazy) vs PySpark on the same workload.

Method
* every (engine, scale, repeat) runs in a fresh Python subprocess;
* wall time covers the whole job: read Parquet -> compute -> materialise result;
  for PySpark the SparkSession start-up is measured separately and excluded;
* peak memory = peak of (RSS of the Python process + RSS of all its child
  processes), sampled every 20 ms - this includes Spark's JVM, which is a child
  of the Python driver and invisible to ``getrusage``;
* correctness: every engine returns checksums (row count, per-column sum and
  non-null count); they are compared with the Polars-lazy reference.
* scale k replicates the real rows k times with shifted ``cert`` (labelled as such).
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import polars as pl

from ..utils.config import Paths
from ..utils.io_utils import md_table, utc_now, write_json
from ..utils.logging_utils import get_logger

log = get_logger("benchmark.processing")


class PeakMemory:
    """Background sampler of RSS(process + children) in MiB."""

    def __init__(self, interval: float = 0.02):
        import psutil
        self.proc = psutil.Process()
        self.interval = interval
        self.peak = 0.0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _rss(self) -> float:
        total = 0
        try:
            total += self.proc.memory_info().rss
            for ch in self.proc.children(recursive=True):
                try:
                    total += ch.memory_info().rss
                except Exception:  # child exited between listing and reading
                    pass
        except Exception:
            pass
        return total / 2**20

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._rss())
            time.sleep(self.interval)

    def __enter__(self):
        self.baseline = self._rss()
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self._rss())


def _worker(spec: dict[str, Any]) -> dict[str, Any]:
    from . import engines
    engine, path, scale = spec["engine"], spec["path"], int(spec["scale"])
    spark, startup = None, 0.0
    if engine == "pyspark":
        t0 = time.perf_counter()
        spark = engines.spark_session(spec.get("spark_driver_memory", "4g"),
                                      int(spec.get("spark_shuffle_partitions", 8)))
        spark.range(1).count()  # force JVM start
        startup = time.perf_counter() - t0
    with PeakMemory() as mem:
        t0 = time.perf_counter()
        if engine == "pyspark":
            res = engines.run_spark(path, scale, spark=spark)
        else:
            res = engines.ENGINES[engine](path, scale)
        secs = time.perf_counter() - t0
    if spark is not None:
        spark.stop()
    return {"seconds": secs, "startup_s": startup, "peak_mb": mem.peak,
            "baseline_mb": mem.baseline, "incr_peak_mb": mem.peak - mem.baseline,
            "checksums": res}


def _run(spec: dict[str, Any]) -> dict[str, Any]:
    src_root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(src_root) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    cmd = [sys.executable, "-m", "fdic_ews.features.processing_benchmark", "--worker",
           json.dumps(spec)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"{spec['engine']} failed:\n{r.stderr[-3000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def _engine_available(name: str) -> bool:
    if name != "pyspark":
        return True
    try:
        import pyspark  # noqa: F401
    except ImportError:
        log.warning("pyspark not installed: skipping (pip install pyspark; needs Java 17+)")
        return False
    return True


def _compare(ref: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    worst_rel, n_mismatch = 0.0, 0
    for k, v in ref.items():
        o = other.get(k)
        if k.endswith("__n") or k == "rows":
            n_mismatch += int(o != v)
        else:
            denom = max(abs(v), 1e-9)
            worst_rel = max(worst_rel, abs((o or 0.0) - v) / denom)
    return {"count_mismatches": n_mismatch, "max_rel_diff_sums": worst_rel}


def run_processing_benchmark(cfg: dict[str, Any]) -> pl.DataFrame:
    paths: Paths = cfg["_paths"]
    b = cfg.get("processing_benchmark", {})
    engines = [e for e in b.get("engines", ["pandas", "polars_eager", "polars_lazy", "pyspark"])
               if _engine_available(e)]
    scales = [int(s) for s in b.get("scale_factors", [1, 4])]
    repeats = int(b.get("repeats", 3))
    path = str(paths.silver / "financials.parquet")
    recs, ref = [], {}
    for k in scales:
        for e in list(engines):
            for r in range(repeats + 1):  # run 0 = warm-up (file cache), excluded
                try:
                    m = _run({"engine": e, "path": path, "scale": k,
                              "spark_driver_memory": b.get("spark_driver_memory", "4g"),
                              "spark_shuffle_partitions": b.get("spark_shuffle_partitions", 8)})
                except RuntimeError as err:
                    if e != "pyspark":
                        raise
                    log.warning("PySpark could not run (Java missing or too old? PySpark 4 "
                                "needs Java 17+). Skipping it. Details: %s", str(err)[-500:])
                    engines.remove(e)
                    break
                cs = m.pop("checksums")
                if e == "polars_lazy" and r == 0:
                    ref[k] = cs
                recs.append({"scale": k, "rows": cs["rows"], "engine": e, "run": r, **m,
                             "_cs": cs})
                log.info("x%d %-13s run %d: %.2fs, peak %.0f MB", k, e, r, m["seconds"],
                         m["peak_mb"])
    for rec in recs:
        rec.update(_compare(ref[rec["scale"]], rec.pop("_cs")))
    raw = pl.DataFrame(recs)
    raw.write_csv(paths.tables / "processing_benchmark_raw.csv")
    summ = (raw.filter(pl.col("run") > 0)
            .group_by("scale", "rows", "engine")
            .agg(seconds_median=pl.col("seconds").median(), seconds_min=pl.col("seconds").min(),
                 seconds_max=pl.col("seconds").max(),
                 spark_startup_s=pl.col("startup_s").median(),
                 peak_mb_median=pl.col("peak_mb").median(),
                 incr_peak_mb_median=pl.col("incr_peak_mb").median(),
                 count_mismatches=pl.col("count_mismatches").max(),
                 max_rel_diff_sums=pl.col("max_rel_diff_sums").max())
            .sort("scale", "seconds_median"))
    pandas_t = summ.filter(pl.col("engine") == "pandas").select(
        "scale", pandas_s=pl.col("seconds_median"))
    if pandas_t.height:
        summ = (summ.join(pandas_t, on="scale", how="left")
                .with_columns(speedup_vs_pandas=pl.col("pandas_s") / pl.col("seconds_median"))
                .drop("pandas_s"))
    summ.write_csv(paths.tables / "processing_benchmark_summary.csv")
    import psutil
    env = {"run_at": utc_now(), "python": sys.version.split()[0], "polars": pl.__version__,
           "platform": platform.platform(), "cpu_count": os.cpu_count(),
           "ram_gb": round(psutil.virtual_memory().total / 2**30, 1), "repeats": repeats}
    try:
        import pandas
        env["pandas"] = pandas.__version__
        import pyspark
        env["pyspark"] = pyspark.__version__
    except ImportError:
        pass
    write_json(paths.tables / "processing_benchmark_env.json", env)
    (paths.tables / "processing_benchmark.md").write_text(
        "# Processing benchmark: same CAMELS workload, four engines (measured)\n\n"
        f"Environment: `{json.dumps(env)}`\n\n"
        "Median wall time (read -> compute -> materialise) over repeated runs in fresh "
        "processes. Peak = RSS of Python + child processes (includes the Spark JVM). "
        "`count_mismatches` / `max_rel_diff_sums` compare each engine's output checksums "
        "with Polars lazy (0 = identical results).\n\n" + md_table(summ.to_dicts()),
        encoding="utf-8")
    _plot(summ, paths)
    return summ


def _plot(summ: pl.DataFrame, paths: Paths) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for e in summ["engine"].unique().sort().to_list():
        s = summ.filter(pl.col("engine") == e).sort("rows")
        axes[0].plot(s["rows"].to_list(), s["seconds_median"].to_list(), marker="o", label=e)
        axes[1].plot(s["rows"].to_list(), s["peak_mb_median"].to_list(), marker="o", label=e)
    axes[0].set_ylabel("Wall time, median (s)")
    axes[1].set_ylabel("Peak memory incl. JVM (MB)")
    for ax in axes:
        ax.set_xlabel("Rows processed")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(paths.figures / "processing_benchmark.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        print(json.dumps(_worker(json.loads(sys.argv[2]))))
