"""Module 1: Silver -> Gold (bank-quarter panel with a forward-looking failure label).

Grain: one row per (cert, repdte). The Gold table is the single input of the
feature pipeline (Module 3) and of the model (Module 5).

Label definition (point-in-time, horizon H quarters, default H = 4):

    target_fail_{H}q(i, t) = 1  if  t < FAILDATE_i <= t + H quarters
                             0  otherwise

Leakage controls built into Gold:

* ``available_date = repdte + publication_lag_days``: a Call Report is not
  public on its report date; downstream code must only use rows whose
  ``available_date`` is before the decision date.
* ``label_observable``: rows whose label window ends after the date the failed
  bank list was downloaded are *right-censored* (a 0 there may become a 1).
* Columns that encode the future (failure date, ``active`` flag, institution
  end date, quarters-to-failure) are listed in :data:`LEAKY_COLUMNS`; the
  Module 2 preprocessing refuses to build features from them.
* Report rows dated on/after a bank's failure date are dropped and counted.
"""
from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from ..utils.config import Paths
from ..utils.io_utils import utc_now, write_delta_safe, write_json
from ..utils.logging_utils import get_logger

log = get_logger("gold")

KEY_COLS = ["cert", "repdte"]
# Information that is only known in the future relative to repdte. Never a feature.
LEAKY_COLUMNS = {"faildate", "failyr", "failed_ever", "quarters_to_failure", "active",
                 "endefymd", "restype", "restype1", "cost", "qbfasset", "qbfdep",
                 "label_observable", "available_date", "fail_name"}
META_COLS = {"repyear", "_bronze_file", "_silver_built_at", "name", "city"}


def target_name(h: int) -> str:
    return f"target_fail_{h}q"


def _failure_list_asof(paths: Paths) -> date:
    """Date the failed-bank list was downloaded (end of observable label window)."""
    mp = paths.bronze / "_manifest.jsonl"
    if mp.exists():
        for line in reversed(mp.read_text(encoding="utf-8").splitlines()):
            e = json.loads(line)
            if e.get("event") == "partition_complete" and e.get("partition") == "failures":
                return date.fromisoformat(e["completed_at"][:10])
    return date.today()


def build_gold(cfg: dict[str, Any]) -> dict[str, Any]:
    paths: Paths = cfg["_paths"]
    h = int(cfg["gold"]["label_horizon_quarters"])
    lag = int(cfg["gold"]["publication_lag_days"])
    tgt = target_name(h)
    asof = _failure_list_asof(paths)

    fin = pl.scan_parquet(paths.silver / "financials.parquet")
    inst_cols = set(pl.read_parquet_schema(paths.silver / "institutions.parquet"))
    inst = (pl.scan_parquet(paths.silver / "institutions.parquet")
            .select([c for c in ("cert", "stalp", "bkclass", "regagnt", "estymd", "active",
                                 "endefymd") if c in inst_cols]))
    fail = (pl.scan_parquet(paths.silver / "failures.parquet")
            .select("cert", "faildate", pl.col("name").alias("fail_name"))
            .filter(pl.col("faildate").is_not_null()))

    panel = (fin.join(inst, on="cert", how="left")
             .join(fail, on="cert", how="left")
             .with_columns(
                 repyear=pl.col("repdte").dt.year(),
                 available_date=pl.col("repdte").dt.offset_by(f"{lag}d"),
                 label_window_end=pl.col("repdte").dt.offset_by(f"{3 * h}mo").dt.month_end(),
                 failed_ever=pl.col("faildate").is_not_null().cast(pl.Int8),
             )
             .with_columns(
                 # number of quarter-ends from repdte to the failure date (0 = same quarter)
                 quarters_to_failure=pl.when(pl.col("faildate").is_not_null()).then(
                     (pl.col("faildate").dt.year() - pl.col("repdte").dt.year()) * 4
                     + pl.col("faildate").dt.quarter() - pl.col("repdte").dt.quarter()),
                 **{tgt: ((pl.col("faildate") > pl.col("repdte"))
                          & (pl.col("faildate") <= pl.col("label_window_end")))
                    .fill_null(False).cast(pl.Int8)},
                 label_observable=(pl.col("label_window_end") <= pl.lit(asof)),
                 bank_age_years=((pl.col("repdte") - pl.col("estymd")).dt.total_days() / 365.25)
                 if "estymd" in inst_cols else pl.lit(None, pl.Float64),
             ))

    post_fail = panel.filter(pl.col("faildate").is_not_null()
                             & (pl.col("repdte") >= pl.col("faildate")))
    post_fail_df = post_fail.select("cert", "repdte", "faildate").collect()
    n_post_fail = post_fail_df.height
    if n_post_fail:  # inspect: should be rare (e.g. reports filed for the failure quarter)
        (post_fail_df.group_by("cert", "faildate")
         .agg(rows=pl.len(), first_repdte=pl.col("repdte").min(),
              last_repdte=pl.col("repdte").max())
         .sort("rows", descending=True)
         .write_csv(paths.tables / "gold_post_failure_rows.csv"))
    gold = (panel.filter(~(pl.col("faildate").is_not_null()
                           & (pl.col("repdte") >= pl.col("faildate"))))
            .drop("label_window_end")
            .select(*KEY_COLS, pl.exclude(KEY_COLS))
            .sort(["repdte", "cert"])
            .collect())

    out = paths.gold / "bank_quarter_panel"
    if out.exists():
        shutil.rmtree(out) if out.is_dir() else out.unlink()
    fmt = cfg["gold"].get("gold_format", "delta")
    if fmt == "delta":
        try:
            if write_delta_safe(gold, out, ["repyear"]):
                log.warning("file system refused Delta's atomic rename (e.g. Google Drive): "
                            "wrote the table locally and copied it to %s", out)
        except ImportError:
            log.warning("deltalake not installed: falling back to Parquet for Gold")
            fmt = "parquet"
    if fmt == "parquet":
        out.mkdir(parents=True, exist_ok=True)
        gold.write_parquet(out / "part-0.parquet", compression="zstd", statistics=True)

    pos_by_year = (gold.group_by("repyear")
                   .agg(rows=pl.len(), banks=pl.col("cert").n_unique(),
                        positives=pl.col(tgt).sum(),
                        observable=pl.col("label_observable").sum())
                   .sort("repyear"))
    pos_by_year.write_csv(paths.tables / "gold_label_counts_by_year.csv")
    stats = {"built_at": utc_now(), "format": fmt, "output": str(out), "rows": gold.height,
             "banks": gold["cert"].n_unique(), "target": tgt, "horizon_quarters": h,
             "positives": int(gold[tgt].sum()), "base_rate": float(gold[tgt].mean()),
             "failed_banks_in_panel": int(gold.filter(pl.col("failed_ever") == 1)["cert"].n_unique()),
             "post_failure_rows_dropped": int(n_post_fail),
             "failure_list_asof": str(asof),
             "censored_rows": int((~gold["label_observable"]).sum())}
    write_json(paths.gold / "_gold_build_stats.json", stats)
    log.info("gold: %s", stats)
    return stats


def read_gold(cfg: dict[str, Any], lazy: bool = True) -> pl.LazyFrame | pl.DataFrame:
    """Read the Gold panel whatever its storage format."""
    paths: Paths = cfg["_paths"]
    out: Path = paths.gold / "bank_quarter_panel"
    lf = pl.scan_delta(str(out)) if (out / "_delta_log").exists() \
        else pl.scan_parquet(out / "*.parquet")
    return lf if lazy else lf.collect()
