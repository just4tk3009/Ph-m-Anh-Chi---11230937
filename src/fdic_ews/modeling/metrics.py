"""Ranking and alert metrics for a rare-event early-warning system.

* PR-AUC (average precision) is the headline metric: with base rates of
  0.06-1.5% ROC-AUC is dominated by the easy negatives (Saito & Rehmsmeier, 2015).
* Recall@top-k%: in every quarter the supervisor can examine only the k% of
  banks with the highest scores; recall is the share of all positive
  bank-quarters that fall inside those per-quarter alert lists.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def top_k_flags(repdte: np.ndarray, score: np.ndarray, k_pct: float) -> np.ndarray:
    """Boolean flag: the row is among the top k% scores of its quarter (ceil, >= 1 bank)."""
    df = pl.DataFrame({"q": repdte, "s": score, "i": np.arange(len(score))})
    df = df.with_columns(
        r=pl.col("s").rank("ordinal", descending=True).over("q"),
        n=pl.len().over("q"))
    df = df.with_columns(flag=pl.col("r") <= (pl.col("n") * k_pct / 100).ceil())
    return df.sort("i")["flag"].to_numpy()


def score_metrics(y: np.ndarray, p: np.ndarray, repdte: np.ndarray,
                  ks: Iterable[float]) -> dict[str, Any]:
    y = np.asarray(y).astype(int)
    out: dict[str, Any] = {"rows": int(len(y)), "positives": int(y.sum()),
                           "base_rate_pct": 100 * float(y.mean()) if len(y) else math.nan}
    has_both = 0 < y.sum() < len(y)
    out["pr_auc"] = float(average_precision_score(y, p)) if has_both else math.nan
    out["roc_auc"] = float(roc_auc_score(y, p)) if has_both else math.nan
    out["brier"] = float(brier_score_loss(y, np.clip(p, 0, 1))) if len(y) else math.nan
    for k in ks:
        f = top_k_flags(repdte, p, float(k))
        tp = int((f & (y == 1)).sum())
        out[f"recall_top{k:g}pct"] = tp / y.sum() if y.sum() else math.nan
        out[f"precision_top{k:g}pct"] = tp / f.sum() if f.sum() else math.nan
    return out


def confusion_at(y: np.ndarray, p: np.ndarray, t: float) -> dict[str, int]:
    y = np.asarray(y).astype(int)
    a = p >= t
    return {"tp": int((a & (y == 1)).sum()), "fp": int((a & (y == 0)).sum()),
            "fn": int((~a & (y == 1)).sum()), "tn": int((~a & (y == 0)).sum())}
