"""Module 3 / Part 3: CAMELS feature pipeline with the Polars Lazy API.

Input : Gold bank-quarter panel (Delta)        -> ``read_gold``
Output: ``data/gold/features.parquet``          (one row per bank-quarter)
        ``reports/tables/feature_dictionary.csv`` (name, pillar, formula, sources)

Three rules guarantee the features use only information available at ``repdte``:

1. **De-cumulation of year-to-date items.** Call Report income items are YTD.
   Quarterly flow = YTD(q) - YTD(q-1) inside the same calendar year, Q1 flow = YTD(Q1).
   If the previous quarter is missing (gap in the series) the flow is null, never
   guessed. Flows are annualised (x4) before dividing by stock variables.
2. **Backward-only, gap-aware windows.** Rolling statistics use
   ``rolling_*_by("repdte", window_size="12mo", closed="right")`` = the current
   quarter and the quarters of the previous 12 months. A row-based window of 4
   rows would silently reach further back when a bank skipped a quarter.
   (PySpark equivalent: ``Window.partitionBy("cert").orderBy(qidx).rangeBetween(-3, 0)``
   on an integer quarter index; ``rowsBetween(-3, 0)`` only when there are no gaps.)
3. **Changes only against the immediately preceding quarter** (QoQ) or the same
   quarter one year earlier (YoY), matched on an integer quarter index.

``verify_no_lookahead`` proves rule 2-3 empirically: features computed on data
truncated at a cut-off date are identical to features computed on the full panel.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl

from ..storage.gold import read_gold
from ..utils.config import Paths, to_date
from ..utils.io_utils import utc_now, write_json
from ..utils.logging_utils import get_logger

log = get_logger("features.camels")

YTD_ITEMS = ["netinc", "intinc", "eintexp", "nonii", "nonix", "elnatr", "ntlnls"]
DYNAMIC_BASE = ["equity_to_assets", "npl_ratio", "roa_ann", "texas_ratio",
                "loans_to_deposits", "liquid_assets_to_assets", "nim_ann"]


def _div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """Safe ratio: null when the denominator is missing or <= 0 (ratio undefined)."""
    return pl.when(den > 0).then(num / den).otherwise(None)


def qidx() -> pl.Expr:
    """Integer quarter index: consecutive quarters differ by exactly 1."""
    return pl.col("repdte").dt.year() * 4 + pl.col("repdte").dt.quarter() - 1


# (name, pillar, formula text, source fields, expression) - the feature dictionary
def _ratio_specs() -> list[tuple[str, str, str, str, pl.Expr]]:
    c = pl.col
    return [
        # C - Capital adequacy
        ("equity_to_assets", "C", "EQ / ASSET", "EQ, ASSET", _div(c("eq"), c("asset"))),
        ("leverage_ratio", "C", "Tier 1 leverage ratio (%) as reported", "RBC1AAJ",
         c("rbc1aaj")),
        ("total_rbc_ratio", "C", "Total risk-based capital ratio (%) as reported", "RBCRWAJ",
         c("rbcrwaj")),
        # A - Asset quality
        ("npl_ratio", "A", "NCLNLS / LNLSGR", "NCLNLS, LNLSGR", _div(c("nclnls"), c("lnlsgr"))),
        ("alll_to_loans", "A", "LNATRES / LNLSGR", "LNATRES, LNLSGR",
         _div(c("lnatres"), c("lnlsgr"))),
        ("alll_coverage", "A", "LNATRES / NCLNLS", "LNATRES, NCLNLS",
         _div(c("lnatres"), c("nclnls"))),
        ("pd30_89_to_assets", "A", "P3ASSET / ASSET", "P3ASSET, ASSET",
         _div(c("p3asset"), c("asset"))),
        ("pd90_to_assets", "A", "P9ASSET / ASSET", "P9ASSET, ASSET",
         _div(c("p9asset"), c("asset"))),
        ("nonaccrual_to_assets", "A", "NAASSET / ASSET", "NAASSET, ASSET",
         _div(c("naasset"), c("asset"))),
        ("ore_to_assets", "A", "ORE / ASSET", "ORE, ASSET", _div(c("ore"), c("asset"))),
        ("nco_rate_ann", "A", "4 x NTLNLS_q / LNLSGR", "NTLNLS (YTD), LNLSGR",
         _div(4 * c("ntlnls_q"), c("lnlsgr"))),
        ("provision_rate_ann", "A", "4 x ELNATR_q / LNLSGR", "ELNATR (YTD), LNLSGR",
         _div(4 * c("elnatr_q"), c("lnlsgr"))),
        ("texas_ratio", "A", "(NCLNLS + ORE) / (EQ + LNATRES)", "NCLNLS, ORE, EQ, LNATRES",
         _div(c("nclnls") + c("ore").fill_null(0), c("eq") + c("lnatres"))),
        # M - Management (efficiency proxies)
        ("efficiency_ratio", "M", "NONIX_q / (INTINC_q - EINTEXP_q + NONII_q)",
         "NONIX, INTINC, EINTEXP, NONII (YTD)",
         _div(c("nonix_q"), c("intinc_q") - c("eintexp_q") + c("nonii_q"))),
        ("nonint_exp_to_assets_ann", "M", "4 x NONIX_q / ASSET", "NONIX (YTD), ASSET",
         _div(4 * c("nonix_q"), c("asset"))),
        # E - Earnings
        ("roa_ann", "E", "4 x NETINC_q / ASSET", "NETINC (YTD), ASSET",
         _div(4 * c("netinc_q"), c("asset"))),
        ("roe_ann", "E", "4 x NETINC_q / EQ (null if EQ <= 0)", "NETINC (YTD), EQ",
         _div(4 * c("netinc_q"), c("eq"))),
        ("nim_ann", "E", "4 x (INTINC_q - EINTEXP_q) / ASSET", "INTINC, EINTEXP (YTD), ASSET",
         _div(4 * (c("intinc_q") - c("eintexp_q")), c("asset"))),
        ("nonint_inc_to_assets_ann", "E", "4 x NONII_q / ASSET", "NONII (YTD), ASSET",
         _div(4 * c("nonii_q"), c("asset"))),
        # L - Liquidity
        ("loans_to_deposits", "L", "LNLSNET / DEP", "LNLSNET, DEP", _div(c("lnlsnet"), c("dep"))),
        ("liquid_assets_to_assets", "L", "(CHBAL + SC) / ASSET", "CHBAL, SC, ASSET",
         _div(c("chbal") + c("sc"), c("asset"))),
        ("cash_to_assets", "L", "CHBAL / ASSET", "CHBAL, ASSET", _div(c("chbal"), c("asset"))),
        ("brokered_to_deposits_log1p", "L", "log(1 + BRO / DEP)", "BRO, DEP",
         _div(c("bro"), c("dep")).log1p()),
        ("has_brokered", "L", "1 if BRO > 0", "BRO", (c("bro") > 0).cast(pl.Int8)),
        ("domestic_dep_share", "L", "DEPDOM / DEP", "DEPDOM, DEP", _div(c("depdom"), c("dep"))),
        # S - Sensitivity to market risk / size
        ("securities_to_assets", "S", "SC / ASSET", "SC, ASSET", _div(c("sc"), c("asset"))),
        ("log_assets", "S", "log(ASSET)", "ASSET", pl.when(c("asset") > 0)
         .then(c("asset").log()).otherwise(None)),
    ]


def add_quarterly_flows(lf: pl.LazyFrame) -> pl.LazyFrame:
    """YTD -> quarterly flows, gap-aware (see module docstring, rule 1)."""
    lf = lf.sort("cert", "repdte").with_columns(_qidx=qidx())
    consecutive = (pl.col("_qidx") - pl.col("_qidx").shift(1).over("cert")) == 1
    is_q1 = pl.col("repdte").dt.quarter() == 1
    return lf.with_columns([
        pl.when(is_q1).then(pl.col(c))
        .when(consecutive).then(pl.col(c) - pl.col(c).shift(1).over("cert"))
        .otherwise(None).alias(f"{c}_q")
        for c in YTD_ITEMS if c in lf.collect_schema().names()
    ])


def add_dynamic_features(lf: pl.LazyFrame, base: list[str]) -> pl.LazyFrame:
    """QoQ change, 4-quarter (12-month) rolling mean / std, YoY asset growth."""
    consecutive = (pl.col("_qidx") - pl.col("_qidx").shift(1).over("cert")) == 1
    exprs = []
    for b in base:
        exprs += [
            pl.when(consecutive).then(pl.col(b) - pl.col(b).shift(1).over("cert"))
            .otherwise(None).alias(f"{b}_qoq"),
            pl.col(b).rolling_mean_by("repdte", window_size="12mo", closed="right")
            .over("cert").alias(f"{b}_ma4"),
            pl.col(b).rolling_std_by("repdte", window_size="12mo", closed="right")
            .over("cert").alias(f"{b}_sd4"),
        ]
    lf = lf.with_columns(exprs)
    # YoY growth matched on quarter index (gap-safe): asset(t) / asset(t-4) - 1
    prev = lf.select("cert", (pl.col("_qidx") + 4).alias("_qidx"),
                     pl.col("asset").alias("_asset_lag4"))
    return (lf.join(prev, on=["cert", "_qidx"], how="left")
            .with_columns(asset_growth_yoy=_div(pl.col("asset"), pl.col("_asset_lag4")) - 1)
            .drop("_asset_lag4"))


def feature_dictionary() -> pl.DataFrame:
    rows = [{"feature": n, "pillar": p, "formula": f, "source_fields": s, "type": "level"}
            for n, p, f, s, _ in _ratio_specs()]
    for b in DYNAMIC_BASE:
        rows += [{"feature": f"{b}_qoq", "pillar": "dyn", "formula": f"{b}(t) - {b}(t-1q)",
                  "source_fields": b, "type": "change"},
                 {"feature": f"{b}_ma4", "pillar": "dyn",
                  "formula": f"mean of {b} over (t-12mo, t]", "source_fields": b,
                  "type": "rolling"},
                 {"feature": f"{b}_sd4", "pillar": "dyn",
                  "formula": f"std of {b} over (t-12mo, t]", "source_fields": b,
                  "type": "rolling"}]
    rows += [{"feature": "asset_growth_yoy", "pillar": "S", "formula": "ASSET(t)/ASSET(t-4q) - 1",
              "source_fields": "ASSET", "type": "change"},
             {"feature": "bank_age_years", "pillar": "M", "formula": "(REPDTE - ESTYMD)/365.25",
              "source_fields": "ESTYMD", "type": "level"}]
    return pl.DataFrame(rows)


def feature_names() -> list[str]:
    return feature_dictionary()["feature"].to_list()


def build_features_lazy(gold: pl.LazyFrame) -> pl.LazyFrame:
    """The complete, lazy feature plan (nothing is computed until ``collect``)."""
    lf = add_quarterly_flows(gold)
    lf = lf.with_columns([e.alias(n) for n, _, _, _, e in _ratio_specs()])
    lf = add_dynamic_features(lf, DYNAMIC_BASE)
    keep_meta = ["cert", "repdte", "repyear", "available_date", "stalp", "bkclass"]
    targets = [c for c in gold.collect_schema().names() if c.startswith("target_")]
    leaky_kept = ["label_observable", "faildate"]  # kept for splitting/evaluation only
    names = gold.collect_schema().names()
    cols = [c for c in keep_meta + targets + leaky_kept if c in names] + feature_names()
    return lf.select([c for c in cols if c in lf.collect_schema().names()]).sort("repdte", "cert")


def verify_no_lookahead(gold: pl.DataFrame, cutoff: date) -> dict[str, Any]:
    """Features for rows <= cutoff must not change when all rows after cutoff are deleted."""
    full = build_features_lazy(gold.lazy()).filter(pl.col("repdte") <= cutoff).collect()
    trunc = build_features_lazy(gold.lazy().filter(pl.col("repdte") <= cutoff)).collect()
    feats = feature_names()
    a = full.sort("cert", "repdte").select(feats)
    b = trunc.sort("cert", "repdte").select(feats)
    max_diff = 0.0
    mismatched = []
    for f in feats:
        x, y = a[f].cast(pl.Float64), b[f].cast(pl.Float64)
        null_mismatch = int((x.is_null() != y.is_null()).sum())
        d = (x - y).abs().max()
        d = 0.0 if d is None else float(d)
        if null_mismatch or d > 1e-12:
            mismatched.append(f)
        max_diff = max(max_diff, d)
    return {"cutoff": str(cutoff), "rows_compared": a.height, "features": len(feats),
            "max_abs_diff": max_diff, "features_changed_by_future_rows": mismatched,
            "passed": not mismatched}


def run_features(cfg: dict[str, Any]) -> dict[str, Any]:
    import time
    paths: Paths = cfg["_paths"]
    t0 = time.perf_counter()
    plan = build_features_lazy(read_gold(cfg, lazy=True))
    (paths.tables / "feature_plan_optimized.txt").write_text(plan.explain(), encoding="utf-8")
    feats = plan.collect()
    secs = time.perf_counter() - t0
    out = paths.gold / "features.parquet"
    feats.write_parquet(out, compression="zstd", statistics=True, row_group_size=100_000)

    fd = feature_dictionary()
    miss = feats.select([pl.col(f).is_null().mean().alias(f) for f in fd["feature"]]) \
        .transpose(include_header=True, header_name="feature", column_names=["null_share"])
    fd = fd.join(miss, on="feature", how="left")
    fd.write_csv(paths.tables / "feature_dictionary.csv")

    gold_df = read_gold(cfg, lazy=False)
    cut = to_date(cfg["split"]["train_end"])
    check = verify_no_lookahead(gold_df, cut)
    pl.DataFrame([{k: (", ".join(v) if isinstance(v, list) else v) for k, v in check.items()}]) \
        .write_csv(paths.tables / "features_lookahead_check.csv")
    stats = {"built_at": utc_now(), "rows": feats.height, "n_features": len(feature_names()),
             "seconds_polars_lazy_end_to_end": round(secs, 2), "output": str(out),
             "lookahead_check": check}
    write_json(paths.gold / "_features_build_stats.json", stats)
    log.info("features: %s", {k: v for k, v in stats.items() if k != "lookahead_check"})
    log.info("look-ahead check: %s", check)
    if not check["passed"]:
        raise AssertionError(f"look-ahead detected in features: {check}")
    return stats
