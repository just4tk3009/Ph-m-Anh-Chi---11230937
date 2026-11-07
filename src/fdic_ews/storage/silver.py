"""Module 1: Bronze -> Silver.

Silver = cleaned, schema-enforced, deduplicated, columnar (Parquet + zstd).
Nothing is *imputed* or *clipped* here: those steps learn parameters from
data and therefore belong inside the train-only sklearn Pipeline (Module 2),
otherwise test-period information would leak into the stored tables.

What Silver does:
1. flatten the FDIC ``{"data": {...}}`` envelope;
2. lower-case column names, enforce explicit dtypes (Int64 / Float64 / Date / String);
3. count values that fail the cast (became null) -> data-quality metric;
4. remove exact duplicate rows and resolve (cert, repdte) key duplicates;
5. sort by (repdte, cert) so Parquet row-group statistics enable predicate pushdown;
6. write build statistics to ``silver/_silver_build_stats.json`` (used by the audit).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from ..utils.config import Paths
from ..utils.io_utils import read_gzip_json, utc_now, write_json
from ..utils.logging_utils import get_logger

log = get_logger("silver")

PARQUET_OPTS = dict(compression="zstd", compression_level=3, statistics=True,
                    row_group_size=100_000)

# Explicit schemas. Every financial amount is Float64 (thousands of USD); ids Int64.
STRING_COLS = {"name", "city", "stalp", "stname", "bkclass", "regagnt", "cityst", "restype",
               "restype1", "savr"}
INT_COLS = {"cert", "active", "failyr"}
DATE_COLS_FIN = {"repdte"}
DATE_COLS_REF = {"estymd", "endefymd", "faildate", "insdate"}


def _load_rows(files: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    rows, src = [], []
    for f in files:
        obj = read_gzip_json(f)
        recs = [r.get("data", r) for r in obj.get("data", [])]  # FDIC {"data": {...}} envelope
        rows.extend(recs)
        src.extend([f.as_posix()] * len(recs))
    return rows, src


def _to_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Build a frame without losing values: stringify everything, then cast explicitly."""
    cols = sorted({k for r in rows for k in r})
    data = {c: [None if r.get(c) is None else str(r.get(c)) for r in rows] for c in cols}
    df = pl.DataFrame(data, schema={c: pl.String for c in cols})
    return df.rename({c: c.lower() for c in df.columns})


def _parse_date(col: str) -> pl.Expr:
    """FDIC uses several date encodings across endpoints: try each one."""
    s = pl.col(col).str.strip_chars()
    return pl.coalesce(
        s.str.strptime(pl.Date, "%Y%m%d", strict=False),
        s.str.strptime(pl.Date, "%m/%d/%Y", strict=False),
        s.str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        s.str.slice(0, 10).str.strptime(pl.Date, "%Y-%m-%d", strict=False),
    ).alias(col)


def _enforce_schema(df: pl.DataFrame, date_cols: set[str]) -> tuple[pl.DataFrame, dict[str, int]]:
    exprs = []
    for c in df.columns:
        if c.startswith("_"):
            exprs.append(pl.col(c))
        elif c in date_cols:
            exprs.append(_parse_date(c))
        elif c in STRING_COLS or c == "id":
            exprs.append(pl.col(c).str.strip_chars())
        elif c in INT_COLS:
            exprs.append(pl.col(c).cast(pl.Float64, strict=False).cast(pl.Int64, strict=False))
        else:
            exprs.append(pl.col(c).cast(pl.Float64, strict=False))
    out = df.select(exprs)
    # cast failures = non-null before, null after
    failures = {c: int(df[c].is_not_null().sum() - out[c].is_not_null().sum())
                for c in df.columns if not c.startswith("_")}
    bad = {c: n for c, n in failures.items() if n > 0}
    for c, n in bad.items():
        if n > 0.5 * max(1, int(df[c].is_not_null().sum())):
            log.warning("column %r: %d of its values failed the cast -> wrong dtype in schema?", c, n)
    return out, bad


def build_financials(paths: Paths) -> dict[str, Any]:
    files = sorted((paths.bronze / "financials").glob("REPDTE=*/page_*.json.gz"))
    if not files:
        raise FileNotFoundError(f"No Bronze financials under {paths.bronze/'financials'}; "
                                "run `python pipeline.py ingest` first.")
    rows, src = _load_rows(files)
    raw = _to_frame(rows).with_columns(pl.Series("_bronze_file", src))
    df, cast_fail = _enforce_schema(raw, DATE_COLS_FIN)
    n_in = df.height

    value_cols = [c for c in df.columns if not c.startswith("_")]
    df_exact = df.unique(subset=value_cols, keep="last", maintain_order=True)
    n_exact_dup = n_in - df_exact.height

    key = ["cert", "repdte"]
    dup_keys = df_exact.filter(pl.len().over(key) > 1)
    n_conflicting = dup_keys.select(key).unique().height
    # Resolution rule: keep the record from the most recently written Bronze page.
    df_final = (df_exact.with_row_index("_row")
                .sort("_row").unique(subset=key, keep="last")
                .drop("_row").sort(["repdte", "cert"]))
    df_final = df_final.with_columns(pl.lit(utc_now()).alias("_silver_built_at"))

    out = paths.silver / "financials.parquet"
    df_final.write_parquet(out, **PARQUET_OPTS)
    stats = {"table": "financials", "bronze_files": len(files), "rows_bronze": n_in,
             "exact_duplicates_removed": n_exact_dup,
             "conflicting_key_duplicates": n_conflicting,
             "rows_silver": df_final.height, "n_banks": df_final["cert"].n_unique(),
             "n_quarters": df_final["repdte"].n_unique(),
             "min_repdte": str(df_final["repdte"].min()),
             "max_repdte": str(df_final["repdte"].max()),
             "cast_failures": cast_fail, "output": str(out)}
    log.info("silver financials: %s", {k: v for k, v in stats.items() if k != "cast_failures"})
    return stats


def build_reference(paths: Paths, name: str) -> dict[str, Any]:
    files = sorted((paths.bronze / name).glob("page_*.json.gz"))
    if not files:
        raise FileNotFoundError(f"No Bronze {name}; run `python pipeline.py ingest` first.")
    rows, src = _load_rows(files)
    df, cast_fail = _enforce_schema(_to_frame(rows).with_columns(pl.Series("_bronze_file", src)),
                                    DATE_COLS_REF)
    n_in = df.height
    value_cols = [c for c in df.columns if not c.startswith("_")]
    df = df.unique(subset=value_cols, keep="last", maintain_order=True)
    stats = {"table": name, "rows_bronze": n_in, "exact_duplicates_removed": n_in - df.height,
             "rows_silver": df.height, "cast_failures": cast_fail}
    if name == "failures":
        # The FDIC list mixes outright failures (RESTYPE = "FAILURE") with open-bank
        # assistance transactions (RESTYPE = "ASSISTANCE"), after which the bank keeps
        # operating and reporting. Only failures define the label.
        n_before = df.height
        if "restype" in df.columns:
            df.filter(pl.col("restype").str.to_uppercase() != "FAILURE").write_parquet(
                paths.silver / "failures_excluded_assistance.parquet", **PARQUET_OPTS)
            n_assist = df.filter(pl.col("restype").str.to_uppercase() != "FAILURE").height
            df = df.filter(pl.col("restype").str.to_uppercase() == "FAILURE")
        else:
            n_assist = 0
        n_after_type = df.height
        df = df.filter(pl.col("cert").is_not_null())
        n_no_cert = n_after_type - df.height
        n_after_cert = df.height
        df = df.sort("faildate").unique("cert", keep="first")  # a CERT fails at most once
        stats.update(assistance_rows_excluded=n_assist, rows_without_cert=n_no_cert,
                     repeated_cert_rows=n_after_cert - df.height, rows_silver=df.height,
                     min_faildate=str(df["faildate"].min()), max_faildate=str(df["faildate"].max()))
    out = paths.silver / f"{name}.parquet"
    df.write_parquet(out, **PARQUET_OPTS)
    stats["output"] = str(out)
    log.info("silver %s: %s", name, {k: v for k, v in stats.items() if k != "cast_failures"})
    return stats


def build_silver(cfg: dict[str, Any]) -> dict[str, Any]:
    paths: Paths = cfg["_paths"]
    stats = {"built_at": utc_now(),
             "financials": build_financials(paths),
             "institutions": build_reference(paths, "institutions"),
             "failures": build_reference(paths, "failures")}
    write_json(paths.silver / "_silver_build_stats.json", stats)
    return stats
