"""Module 3: one feature workload, four engines (Pandas, Polars eager, Polars lazy, PySpark).

The workload is a representative slice of the CAMELS pipeline, written so that
all four engines compute *exactly* the same numbers:

1. read 12 columns of ``silver/financials.parquet``;
2. de-cumulate 3 year-to-date items (NETINC, INTINC, EINTEXP), gap-aware;
3. 5 ratios with safe division (null if denominator <= 0);
4. per bank: QoQ change and 4-quarter rolling mean / std of each ratio
   (row-based window of the current + 3 previous reports, min 1 value for the
   mean and 2 for the std - identical semantics in every engine);
5. materialise the result and return checksums (row count, per-column sum and
   non-null count) used to prove that the engines agree.

Row-based windows are used here only because they exist in all four APIs with
identical semantics; the production pipeline (``camels.py``) uses time-based,
gap-aware windows.
"""
from __future__ import annotations

from typing import Any

READ_COLS = ["cert", "repdte", "asset", "dep", "eq", "lnlsgr", "lnlsnet", "nclnls",
             "netinc", "intinc", "eintexp", "chbal"]
YTD = ["netinc", "intinc", "eintexp"]
RATIOS = ["equity_to_assets", "npl_ratio", "roa_ann", "nim_ann", "loans_to_deposits"]
OUT_COLS = RATIOS + [f"{r}_{s}" for r in RATIOS for s in ("qoq", "ma4", "sd4")]


def _checksums_from_polars(df) -> dict[str, Any]:
    import polars as pl
    out = {"rows": df.height}
    for c in OUT_COLS:
        s = df[c].cast(pl.Float64)
        out[f"{c}__sum"] = float(s.sum() or 0.0)
        out[f"{c}__n"] = int(s.is_not_null().sum())
    return out


# --------------------------------------------------------------------------- Polars
def _polars_plan(lf):
    """Works on a LazyFrame (optimised plan) or a DataFrame (eager, step by step)."""
    import polars as pl

    def div(a, b):
        return pl.when(b > 0).then(a / b).otherwise(None)

    lf = (lf.sort("cert", "repdte")
          .with_columns(_q=pl.col("repdte").dt.year() * 4 + pl.col("repdte").dt.quarter()))
    consec = (pl.col("_q") - pl.col("_q").shift(1).over("cert")) == 1
    q1 = pl.col("repdte").dt.quarter() == 1
    lf = lf.with_columns([pl.when(q1).then(pl.col(c)).when(consec)
                          .then(pl.col(c) - pl.col(c).shift(1).over("cert"))
                          .otherwise(None).alias(f"{c}_q") for c in YTD])
    lf = lf.with_columns(
        equity_to_assets=div(pl.col("eq"), pl.col("asset")),
        npl_ratio=div(pl.col("nclnls"), pl.col("lnlsgr")),
        roa_ann=div(4 * pl.col("netinc_q"), pl.col("asset")),
        nim_ann=div(4 * (pl.col("intinc_q") - pl.col("eintexp_q")), pl.col("asset")),
        loans_to_deposits=div(pl.col("lnlsnet"), pl.col("dep")),
    )
    exprs = []
    for r in RATIOS:
        exprs += [pl.when(consec).then(pl.col(r) - pl.col(r).shift(1).over("cert"))
                  .otherwise(None).alias(f"{r}_qoq"),
                  pl.col(r).rolling_mean(4, min_samples=1).over("cert").alias(f"{r}_ma4"),
                  pl.col(r).rolling_std(4, min_samples=2).over("cert").alias(f"{r}_sd4")]
    return lf.with_columns(exprs).select("cert", "repdte", *OUT_COLS)


def run_polars_lazy(path: str, scale: int = 1) -> dict[str, Any]:
    import polars as pl
    lf = pl.scan_parquet(path).select(READ_COLS)  # projection pushed into the reader
    lf = _replicate_polars(lf, scale)
    return _checksums_from_polars(_polars_plan(lf).collect())


def run_polars_eager(path: str, scale: int = 1) -> dict[str, Any]:
    import polars as pl
    df = pl.read_parquet(path, columns=READ_COLS)
    if scale > 1:
        df = pl.concat([df.with_columns(pl.col("cert") + i * 10_000_000) for i in range(scale)])
    # identical expressions on a DataFrame: every step is materialised immediately
    # (no query optimiser, no projection/predicate pushdown, no common-subplan reuse)
    return _checksums_from_polars(_polars_plan(df))


def _replicate_polars(lf, scale: int):
    import polars as pl
    if scale == 1:
        return lf
    off = 10_000_000
    return pl.concat([lf.with_columns(pl.col("cert") + i * off) for i in range(scale)])


# --------------------------------------------------------------------------- Pandas
def run_pandas(path: str, scale: int = 1) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    df = pd.read_parquet(path, columns=READ_COLS)
    if scale > 1:
        df = pd.concat([df.assign(cert=df["cert"] + i * 10_000_000) for i in range(scale)],
                       ignore_index=True)
    df["repdte"] = pd.to_datetime(df["repdte"])
    num = [c for c in READ_COLS if c not in ("cert", "repdte")]
    df[num] = df[num].astype("float64")
    df = df.sort_values(["cert", "repdte"], kind="mergesort").reset_index(drop=True)
    df["_q"] = df["repdte"].dt.year * 4 + df["repdte"].dt.quarter
    g = df.groupby("cert", sort=False)
    consec = (df["_q"] - g["_q"].shift(1)) == 1
    q1 = df["repdte"].dt.quarter == 1
    for c in YTD:
        prev = g[c].shift(1)
        df[f"{c}_q"] = np.where(q1, df[c], np.where(consec, df[c] - prev, np.nan))

    def div(a, b):
        return np.where(b > 0, a / b.where(b > 0), np.nan)

    df["equity_to_assets"] = div(df["eq"], df["asset"])
    df["npl_ratio"] = div(df["nclnls"], df["lnlsgr"])
    df["roa_ann"] = div(4 * df["netinc_q"], df["asset"])
    df["nim_ann"] = div(4 * (df["intinc_q"] - df["eintexp_q"]), df["asset"])
    df["loans_to_deposits"] = div(df["lnlsnet"], df["dep"])
    g = df.groupby("cert", sort=False)
    for r in RATIOS:
        df[f"{r}_qoq"] = np.where(consec, df[r] - g[r].shift(1), np.nan)
        roll = g[r].rolling(4, min_periods=1)
        df[f"{r}_ma4"] = roll.mean().reset_index(level=0, drop=True)
        df[f"{r}_sd4"] = g[r].rolling(4, min_periods=2).std().reset_index(level=0, drop=True)
    out = {"rows": len(df)}
    for c in OUT_COLS:
        s = df[c].astype("float64")
        out[f"{c}__sum"] = float(np.nansum(s.to_numpy()))
        out[f"{c}__n"] = int(s.notna().sum())
    return out


# --------------------------------------------------------------------------- PySpark
def spark_session(driver_memory: str = "4g", shuffle_partitions: int = 8):
    from pyspark.sql import SparkSession
    return (SparkSession.builder.master("local[*]").appName("bft65b-camels")
            .config("spark.driver.memory", driver_memory)
            .config("spark.sql.shuffle.partitions", shuffle_partitions)  # default 200 is for clusters
            .config("spark.ui.enabled", "false")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate())


def spark_plan(sdf):
    """PySpark equivalent of ``_polars_plan`` (Window functions, backward-only)."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    def div(a, b):
        return F.when(b > 0, a / b)

    w = Window.partitionBy("cert").orderBy("repdte")
    w4 = w.rowsBetween(-3, 0)  # current + 3 previous reports (NOT -4, which is 5 rows)
    sdf = sdf.withColumn("_q", F.year("repdte") * 4 + F.quarter("repdte"))
    consec = (F.col("_q") - F.lag("_q", 1).over(w)) == 1
    q1 = F.quarter("repdte") == 1
    for c in YTD:
        sdf = sdf.withColumn(f"{c}_q", F.when(q1, F.col(c))
                             .when(consec, F.col(c) - F.lag(c, 1).over(w)))
    sdf = (sdf.withColumn("equity_to_assets", div(F.col("eq"), F.col("asset")))
              .withColumn("npl_ratio", div(F.col("nclnls"), F.col("lnlsgr")))
              .withColumn("roa_ann", div(4 * F.col("netinc_q"), F.col("asset")))
              .withColumn("nim_ann", div(4 * (F.col("intinc_q") - F.col("eintexp_q")),
                                         F.col("asset")))
              .withColumn("loans_to_deposits", div(F.col("lnlsnet"), F.col("dep"))))
    for r in RATIOS:
        sdf = (sdf.withColumn(f"{r}_qoq", F.when(consec, F.col(r) - F.lag(r, 1).over(w)))
                  .withColumn(f"{r}_ma4", F.avg(r).over(w4))
                  .withColumn(f"{r}_sd4", F.when(F.count(r).over(w4) >= 2,
                                                 F.stddev_samp(r).over(w4))))
    return sdf.select("cert", "repdte", *OUT_COLS)


def run_spark(path: str, scale: int = 1, spark=None) -> dict[str, Any]:
    from pyspark.sql import functions as F
    own = spark is None
    spark = spark or spark_session()
    sdf = spark.read.parquet(path).select(*READ_COLS)
    for c in READ_COLS[2:]:
        sdf = sdf.withColumn(c, F.col(c).cast("double"))
    if scale > 1:
        base = sdf
        for i in range(1, scale):
            sdf = sdf.unionByName(base.withColumn("cert", F.col("cert") + i * 10_000_000))
    res = spark_plan(sdf)
    aggs = [F.count(F.lit(1)).alias("rows")]
    for c in OUT_COLS:
        aggs += [F.sum(c).alias(f"{c}__sum"), F.count(c).alias(f"{c}__n")]
    row = res.agg(*aggs).collect()[0].asDict()
    if own:
        spark.stop()
    return {k: (float(v) if k.endswith("__sum") else int(v or 0)) if v is not None else 0.0
            for k, v in row.items()}


ENGINES = {"pandas": run_pandas, "polars_eager": run_polars_eager,
           "polars_lazy": run_polars_lazy, "pyspark": run_spark}
