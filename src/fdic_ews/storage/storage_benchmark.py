"""Module 1 / Part 1 proof: CSV vs Parquet (snappy, zstd) vs Delta Lake.

The same Silver financials table is written in each format and three typical
access patterns are timed:

* ``full_scan``   - read every column and row (worst case);
* ``projection``  - read 4 columns only (columnar formats skip the rest);
* ``filter_proj`` - 4 columns for a single year (predicate pushdown / partition pruning).

Method (so the numbers can be defended in the report):

* every measurement runs in a **fresh Python subprocess**, so OS page cache is
  the only thing shared between runs and Python heap from a previous run cannot
  distort memory figures;
* **peak RSS** is read from the OS (Linux: ``VmHWM`` after resetting it via
  ``/proc/self/clear_refs`` once imports are done, i.e. exact; macOS/Windows:
  ``getrusage`` / ``psutil`` peak minus RSS after imports, an approximation)
  and reported as the increase over the post-import baseline;
* each measurement is repeated ``benchmark.repeats`` times; the median and the
  min-max range are reported; the first (cold-cache) run is kept separately;
* ``scale_factors`` > 1 replicate the real rows (with shifted ``cert``) to show
  how read time and memory grow with size. Those rows are *replicated real
  data*, and the report must say so. Replicated rows compress unrealistically
  well, so cite file-size ratios from the x1 (real) run only.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import polars as pl

from ..utils.config import Paths
from ..utils.io_utils import folder_size_bytes, md_table, utc_now, write_json
from ..utils.logging_utils import get_logger

log = get_logger("benchmark.storage")

FORMATS = ["csv", "parquet_snappy", "parquet_zstd", "delta_zstd"]
QUERIES = ["full_scan", "projection", "filter_proj"]
PROJ_COLS = ["cert", "repdte", "asset", "nclnls"]


# ------------------------------------------------------------------ writers
def _target(root: Path, fmt: str) -> Path:
    return root / {"csv": "data.csv", "parquet_snappy": "data_snappy.parquet",
                   "parquet_zstd": "data_zstd.parquet", "delta_zstd": "delta"}[fmt]


def write_format(df: pl.DataFrame, fmt: str, root: Path) -> tuple[Path, float]:
    path = _target(root, fmt)
    if path.exists():
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    t0 = time.perf_counter()
    if fmt == "csv":
        df.write_csv(path)
    elif fmt == "parquet_snappy":
        df.write_parquet(path, compression="snappy", statistics=True, row_group_size=100_000)
    elif fmt == "parquet_zstd":
        df.write_parquet(path, compression="zstd", compression_level=3, statistics=True,
                         row_group_size=100_000)
    elif fmt == "delta_zstd":
        df.write_delta(str(path), mode="overwrite",
                       delta_write_options={"partition_by": ["repyear"]})
    return path, time.perf_counter() - t0


# ------------------------------------------------------------------ worker (subprocess)
def _reset_peak() -> bool:
    """Linux only: reset the kernel's peak-RSS counter (VmHWM) so imports are excluded."""
    try:
        with open("/proc/self/clear_refs", "w") as fh:
            fh.write("5")
        return True
    except OSError:
        return False


def _rss_mb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 2**20


def _peak_rss_mb() -> float:
    if platform.system() == "Linux":
        for line in open("/proc/self/status", encoding="ascii"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 2**10  # KiB -> MiB
    if platform.system() == "Windows":
        import psutil
        return psutil.Process().memory_info().peak_wset / 2**20
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if platform.system() == "Darwin" else peak / 2**10


def _worker(spec: dict[str, Any]) -> dict[str, Any]:
    """Executed inside a fresh interpreter: run one query once and report time + memory.

    Memory: on Linux the peak counter is reset after imports, so ``incr_peak_mb``
    is exactly the extra peak memory of the query. On macOS/Windows the peak
    since process start is used minus the RSS after imports (an upper-bound
    approximation; say so in the report if you benchmark there).
    """
    import polars as pl  # noqa: F401  (import cost is outside the timed region)

    fmt, query, path, year = spec["fmt"], spec["query"], spec["path"], spec["year"]
    if fmt.startswith("delta"):
        import deltalake  # noqa: F401  (pl.scan_delta imports it lazily: keep it untimed)
    exact = _reset_peak()
    base = _rss_mb()
    t0 = time.perf_counter()
    if fmt == "csv":
        lf = pl.scan_csv(path, try_parse_dates=True)
    elif fmt.startswith("parquet"):
        lf = pl.scan_parquet(path)
    else:
        lf = pl.scan_delta(path)
    if query in ("projection", "filter_proj"):
        lf = lf.select(PROJ_COLS + (["repyear"] if query == "filter_proj" else []))
    if query == "filter_proj":
        lf = lf.filter(pl.col("repyear") == year)
    df = lf.collect()
    secs = time.perf_counter() - t0
    peak = _peak_rss_mb()
    return {"seconds": secs, "peak_rss_mb": peak, "baseline_rss_mb": base,
            "incr_peak_mb": max(0.0, peak - base), "result_mb": df.estimated_size("mb"),
            "rows": df.height, "cols": df.width, "peak_reset_exact": exact}


def _run_subprocess(spec: dict[str, Any]) -> dict[str, Any]:
    src_root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(src_root) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    cmd = [sys.executable, "-m", "fdic_ews.storage.storage_benchmark", "--worker", json.dumps(spec)]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"benchmark worker failed: {res.stderr[-2000:]}")
    return json.loads(res.stdout.strip().splitlines()[-1])


# ------------------------------------------------------------------ driver
def _scaled(df: pl.DataFrame, k: int) -> pl.DataFrame:
    if k == 1:
        return df
    offset = int(df["cert"].max()) + 1
    return pl.concat([df.with_columns(pl.col("cert") + i * offset) for i in range(k)])


def run_storage_benchmark(cfg: dict[str, Any]) -> pl.DataFrame:
    paths: Paths = cfg["_paths"]
    repeats = int(cfg["benchmark"]["repeats"])
    scales = [int(s) for s in cfg["benchmark"]["scale_factors"]]
    base = (pl.read_parquet(paths.silver / "financials.parquet")
            .drop("_bronze_file", "_silver_built_at", strict=False)
            .with_columns(repyear=pl.col("repdte").dt.year()))
    years = base["repyear"].unique().sort().to_list()
    year = years[len(years) // 2]  # a mid-sample year for the filtered query

    records: list[dict[str, Any]] = []
    for k in scales:
        df = _scaled(base, k)
        root = paths.data / "_benchmark" / f"x{k}"
        root.mkdir(parents=True, exist_ok=True)
        for fmt in FORMATS:
            path, wsec = write_format(df, fmt, root)
            size_mb = folder_size_bytes(path) / 2**20
            log.info("x%d %-15s size=%.1f MB write=%.2fs", k, fmt, size_mb, wsec)
            for q in QUERIES:
                for r in range(repeats + 1):  # run 0 = cold-ish, excluded from median
                    m = _run_subprocess({"fmt": fmt, "query": q, "path": str(path), "year": year})
                    records.append({"scale": k, "rows_table": df.height, "format": fmt,
                                    "query": q, "run": r, "size_mb": size_mb,
                                    "write_s": wsec, **m})
    raw = pl.DataFrame(records)
    raw.write_csv(paths.tables / "storage_benchmark_raw.csv")

    summ = (raw.filter(pl.col("run") > 0)
            .group_by(["scale", "rows_table", "format", "query"])
            .agg(size_mb=pl.col("size_mb").first(), write_s=pl.col("write_s").first(),
                 read_s_median=pl.col("seconds").median(), read_s_min=pl.col("seconds").min(),
                 read_s_max=pl.col("seconds").max(),
                 incr_peak_mb_median=pl.col("incr_peak_mb").median(),
                 result_mb=pl.col("result_mb").first(),
                 rows_out=pl.col("rows").first(), cols_out=pl.col("cols").first())
            .sort(["scale", "query", "format"]))
    # ratios vs CSV for the same scale and query = the number the report should cite
    csv_ref = summ.filter(pl.col("format") == "csv").select(
        "scale", "query", csv_size=pl.col("size_mb"), csv_read=pl.col("read_s_median"))
    summ = (summ.join(csv_ref, on=["scale", "query"])
            .with_columns(size_vs_csv=pl.col("size_mb") / pl.col("csv_size"),
                          speedup_vs_csv=pl.col("csv_read") / pl.col("read_s_median"))
            .drop("csv_size", "csv_read"))
    summ.write_csv(paths.tables / "storage_benchmark_summary.csv")

    env = {"run_at": utc_now(), "python": sys.version.split()[0], "polars": pl.__version__,
           "platform": platform.platform(), "cpu_count": os.cpu_count(),
           "repeats": repeats, "filter_year": year}
    write_json(paths.tables / "storage_benchmark_env.json", env)
    md = ["# Storage-format benchmark (measured)\n",
          f"Environment: `{json.dumps(env)}`\n",
          "Median of repeated runs, each in a fresh process. `incr_peak_mb` = peak RSS "
          "above the interpreter baseline.\n",
          md_table(summ.select("scale", "rows_table", "format", "query", "size_mb",
                               "size_vs_csv", "read_s_median", "speedup_vs_csv",
                               "incr_peak_mb_median", "result_mb").to_dicts())]
    (paths.tables / "storage_benchmark.md").write_text("\n".join(md), encoding="utf-8")
    _plot(summ, paths)
    return summ


def _plot(summ: pl.DataFrame, paths: Paths) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    s1 = summ.filter(pl.col("scale") == summ["scale"].min())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    sizes = s1.filter(pl.col("query") == "full_scan")
    axes[0].barh(sizes["format"].to_list(), sizes["size_mb"].to_list(), color="#4c72b0")
    axes[0].set_xlabel("Size on disk (MB)")
    axes[0].set_title("Storage footprint")
    width = 0.27
    fmts = FORMATS
    for i, q in enumerate(QUERIES):
        sub = s1.filter(pl.col("query") == q)
        vals = [sub.filter(pl.col("format") == f)["read_s_median"].item() for f in fmts]
        axes[1].bar([j + (i - 1) * width for j in range(len(fmts))], vals, width, label=q)
    axes[1].set_xticks(range(len(fmts)), fmts, rotation=15)
    axes[1].set_ylabel("Read time, median (s)")
    axes[1].set_title("Read latency by access pattern")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(paths.figures / "storage_benchmark.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        print(json.dumps(_worker(json.loads(sys.argv[2]))))
