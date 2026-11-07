"""Part 6: score one quarter with the frozen, serialised model.

Loads ``data/models/ews_model.joblib`` (pipeline + Platt calibrator + threshold
+ feature list, written by ``pipeline.py model``) and scores every bank of one
quarter of ``gold/features.parquet``. Default: the latest quarter in the data,
whose 4-quarter outcome is not yet known - the real use of an early-warning
system. No refitting happens here.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import polars as pl

from ..utils.config import Paths
from ..utils.io_utils import md_table, utc_now, write_json
from ..utils.logging_utils import get_logger
from .metrics import top_k_flags

log = get_logger("modeling.predict")


def _quarter_end(q: str) -> date:
    y, n = int(q[:4]), int(q[-1])
    return {1: date(y, 3, 31), 2: date(y, 6, 30), 3: date(y, 9, 30), 4: date(y, 12, 31)}[n]


def run_predict(cfg: dict[str, Any], quarter: str | None = None, top_n: int = 25) -> dict[str, Any]:
    import joblib
    paths: Paths = cfg["_paths"]
    bundle = joblib.load(paths.data / "models" / "ews_model.joblib")
    feats = bundle["features"]
    data = pl.read_parquet(paths.gold / "features.parquet")
    q = _quarter_end(quarter) if quarter else data["repdte"].max()
    rows = data.filter(pl.col("repdte") == q)
    if rows.is_empty():
        raise ValueError(f"no bank-quarters for {q}")
    raw = bundle["pipeline"].predict_proba(rows.select(feats).to_pandas())[:, 1]
    p = bundle["calibrator"].transform(raw)
    k = float(cfg["model"]["recall_top_k_pct"][1])
    out = (rows.select([c for c in ("cert", "repdte", "stalp", "bkclass") if c in rows.columns])
           .with_columns(prob_fail_4q=pl.Series(p),
                         alert=pl.Series(p >= bundle["threshold"]),
                         top_k=pl.Series(top_k_flags(rows["repdte"].to_numpy(), p, k)))
           .sort("prob_fail_4q", descending=True)
           .with_columns(rank=pl.int_range(1, pl.len() + 1)))
    out.write_csv(paths.data / "models" / f"scores_{q}.csv")
    out.head(top_n).write_csv(paths.tables / "predict_latest_quarter_top.csv")
    summ = {"scored_at": utc_now(), "quarter": str(q), "banks_scored": out.height,
            "model": bundle["choice"]["chosen_model"], "threshold": bundle["threshold"],
            "alerts_at_threshold": int(out["alert"].sum()),
            f"alerts_top{k:g}pct": int(out["top_k"].sum()),
            "mean_prob_pct": 100 * float(np.mean(p))}
    write_json(paths.tables / "predict_latest_quarter_summary.json", summ)
    (paths.tables / "predict_latest_quarter.md").write_text(
        f"# Early-warning scores for {q} (generated {utc_now()})\n\n"
        "Frozen model from `pipeline.py model`; outcome not yet observable.\n\n"
        + md_table([summ]) + "\n\nHighest-risk banks:\n\n" + md_table(out.head(top_n).to_dicts()),
        encoding="utf-8")
    log.info("predict: %s", summ)
    return summ
