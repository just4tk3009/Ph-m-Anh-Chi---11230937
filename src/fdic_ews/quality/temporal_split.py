"""Leakage-safe temporal split with purging (Module 2, reused by Modules 4-5).

Rule: *split first, transform later*. Every learned preprocessing parameter
(medians, IQR fences, scales, selected columns) is fitted on the train window
only, after this split.

Purging: a train row dated ``t`` has a label that looks ``H`` quarters ahead,
so its label is only known at ``t + H``. If ``t + H`` falls after the train
cut-off, the label uses outcomes from the validation period, so the row is
dropped (López de Prado, 2018, ch. 7). The same rule applies to the
validation window, and test rows must have a fully observed label window.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import polars as pl

from ..storage.gold import target_name
from ..utils.config import to_date


@dataclass
class TemporalSplit:
    train: pl.DataFrame
    valid: pl.DataFrame
    test: pl.DataFrame
    stats: list[dict[str, Any]] = field(default_factory=list)


def _window_end(col: str, h: int) -> pl.Expr:
    return pl.col(col).dt.offset_by(f"{3 * h}mo").dt.month_end()


def temporal_split(gold: pl.DataFrame, cfg: dict[str, Any]) -> TemporalSplit:
    s = cfg["split"]
    h = int(cfg["gold"]["label_horizon_quarters"])
    tgt = target_name(h)
    t0, t1 = to_date(s["train_start"]), to_date(s["train_end"])
    v1, e1 = to_date(s["valid_end"]), to_date(s["test_end"])
    purge = bool(s.get("purge_horizon", True))

    df = gold.with_columns(_lw_end=_window_end("repdte", h))
    windows: dict[str, tuple[date, date, date]] = {   # (start excl/incl, end, label must end by)
        "train": (t0, t1, t1), "valid": (t1, v1, v1), "test": (v1, e1, e1)}
    out, stats = {}, []
    for name, (lo, hi, label_by) in windows.items():
        in_window = (pl.col("repdte") >= lo) if name == "train" else (pl.col("repdte") > lo)
        in_window &= pl.col("repdte") <= hi
        part = df.filter(in_window)
        n_window = part.height
        if purge and name != "test":
            part = part.filter(pl.col("_lw_end") <= label_by)
        if name == "test" and "label_observable" in part.columns:
            part = part.filter(pl.col("label_observable"))
        stats.append({"split": name, "from": str(lo), "to": str(hi), "rows_in_window": n_window,
                      "rows_purged": n_window - part.height, "rows": part.height,
                      "banks": part["cert"].n_unique(),
                      "positives": int(part[tgt].sum()) if part.height else 0,
                      "base_rate_pct": 100 * float(part[tgt].mean()) if part.height else 0.0,
                      "max_label_window_end": str(part["_lw_end"].max()) if part.height else ""})
        out[name] = part.drop("_lw_end")
    return TemporalSplit(out["train"], out["valid"], out["test"], stats)


def assert_no_temporal_overlap(split: TemporalSplit, h: int) -> None:
    """Hard check used by tests and the audit: no train label window reaches the valid period."""
    for a, b, name in ((split.train, split.valid, "train->valid"),
                       (split.valid, split.test, "valid->test")):
        if a.height and b.height:
            last_label_end = a.select(_window_end("repdte", h).max()).item()
            first_next = b["repdte"].min()
            assert a["repdte"].max() < first_next, f"{name}: feature windows overlap"
            assert last_label_end < first_next, (
                f"{name}: a label window ends {last_label_end}, after the next split starts "
                f"{first_next} (enable split.purge_horizon)")
