"""Misclassification costs estimated from FDIC data, and the cost-minimising threshold.

CFN (missed failure) comes from the data: the FDIC's estimated loss to the
deposit-insurance fund (``COST`` in the failures table, $ thousands) of the
failures whose label falls in the development period. Only a share ``alpha``
of that loss is assumed avoidable by an earlier supervisory action, and a
failing bank contributes several positive bank-quarters, so the cost of one
missed positive row is

    CFN_row = alpha * median(COST of development failures) / mean positive rows per failed bank

so that missing a failing bank in every one of its positive quarters costs
alpha * median loss. CFP (false alarm = one targeted examination) has no
counterpart in the FDIC data and is an explicit assumption; both enter a
sensitivity table.

Threshold rule. ``t* = CFP / (CFP + CFN)`` is Bayes-optimal only for
calibrated probabilities. Probabilities are therefore Platt-calibrated on the
validation window, and the threshold actually used is the one that minimises
the realised cost on that window; ``t*`` is reported next to it.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import polars as pl

from ..utils.config import Paths


def estimate_costs(paths: Paths, dev_from: date, label_by: date, pos_rows_per_failed_bank: float,
                   alpha: float, cfp: float) -> tuple[dict[str, Any], pl.DataFrame]:
    f = pl.read_parquet(paths.silver / "failures.parquet")
    dev = f.filter(pl.col("faildate").is_between(dev_from, label_by))
    with_cost = dev.filter(pl.col("cost").is_not_null() & (pl.col("cost") >= 0))
    rate = with_cost.filter(pl.col("qbfasset") > 0).with_columns(
        loss_rate=pl.col("cost") / pl.col("qbfasset"))["loss_rate"]
    med_cost = float(with_cost["cost"].median()) if with_cost.height else float("nan")
    cfn_bank = alpha * med_cost
    cfn_row = cfn_bank / max(pos_rows_per_failed_bank, 1.0)
    stats = {"failures_in_development": dev.height,
             "failures_with_cost": with_cost.height,
             "cost_median_thousand_usd": med_cost,
             "cost_mean_thousand_usd": float(with_cost["cost"].mean()) if with_cost.height else None,
             "cost_p25_thousand_usd": float(with_cost["cost"].quantile(0.25)) if with_cost.height else None,
             "cost_p75_thousand_usd": float(with_cost["cost"].quantile(0.75)) if with_cost.height else None,
             "loss_rate_median_pct_of_assets": 100 * float(rate.median()) if rate.len() else None,
             "avoidable_loss_share_alpha": alpha,
             "positive_rows_per_failed_bank": pos_rows_per_failed_bank,
             "cfn_per_missed_bank_thousand_usd": cfn_bank,
             "cfn_per_positive_row_thousand_usd": cfn_row,
             "cfp_per_false_alarm_thousand_usd": cfp,
             "cfn_over_cfp": cfn_row / cfp,
             "bayes_threshold_calibrated": cfp / (cfp + cfn_row),
             "dollars": "nominal (not inflation-adjusted)"}
    by_year = (with_cost.group_by(pl.col("faildate").dt.year().alias("fail_year"))
               .agg(failures=pl.len(), cost_median=pl.col("cost").median(),
                    cost_sum=pl.col("cost").sum()).sort("fail_year"))
    return stats, by_year


def cost_curve(y: np.ndarray, p: np.ndarray, cfn: float, cfp: float,
               n_grid: int = 400) -> pl.DataFrame:
    """Realised cost FN*CFN + FP*CFP over candidate thresholds (quantiles of p, plus 0 and 1)."""
    y = np.asarray(y).astype(int)
    qs = np.unique(np.concatenate([[0.0, 1.0 + 1e-12],
                                   np.quantile(p, np.linspace(0, 1, n_grid)),
                                   np.quantile(p[y == 1], np.linspace(0, 1, 50)) if y.sum() else []]))
    order = np.argsort(-p)
    ps, ys = p[order], y[order]
    tp_cum = np.cumsum(ys)
    fp_cum = np.cumsum(1 - ys)
    P = int(y.sum())
    rows = []
    for t in qs:
        n = int(np.searchsorted(-ps, -t, side="right"))  # rows with p >= t
        tp = int(tp_cum[n - 1]) if n else 0
        fp = int(fp_cum[n - 1]) if n else 0
        fn = P - tp
        rows.append({"threshold": float(t), "alerts": n, "tp": tp, "fp": fp, "fn": fn,
                     "cost_thousand_usd": fn * cfn + fp * cfp})
    return pl.DataFrame(rows).sort("threshold")


def best_threshold(y: np.ndarray, p: np.ndarray, cfn: float, cfp: float) -> tuple[float, pl.DataFrame]:
    cc = cost_curve(y, p, cfn, cfp)
    best = cc.sort(["cost_thousand_usd", "threshold"], descending=[False, True]).row(0, named=True)
    return float(best["threshold"]), cc
