"""Module 2 / Part 2: data-quality audit of the Silver/Gold tables.

Produces measured evidence for every cleaning decision. Outputs:

* ``reports/tables/dq_*.csv``  - one table per check (cite them in the report);
* ``reports/figures/dq_*.png`` - missingness heat-map, outlier comparison;
* ``reports/module2_data_quality.md`` - all tables in one readable file.

Checks
1. Lineage and duplicates (from the Silver build statistics + re-check).
2. Panel coverage: banks per quarter, gaps inside a bank's quarterly series.
3. Schema profile: dtype, nulls, distinct values, min/max per column.
4. Missingness: overall, by year, by asset-size quintile and by label
   -> evidence for an MCAR / MAR / MNAR / structural classification.
5. Zero- and near-zero-variance columns, decided on the TRAIN window only.
6. Accounting-identity / domain-rule violations.
7. Outliers on core ratios: Tukey IQR fences (fitted on train) vs mean +/- 3 sd,
   skewness/kurtosis, and the failure rate among flagged rows.
8. Multivariate outliers: Local Outlier Factor fitted on a train sample,
   applied (novelty mode) to validation and test.
9. Label distribution per split (class imbalance).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ..storage.gold import KEY_COLS, LEAKY_COLUMNS, META_COLS, read_gold, target_name
from ..utils.config import Paths
from ..utils.io_utils import md_table, read_json, utc_now
from ..utils.logging_utils import get_logger
from .temporal_split import assert_no_temporal_overlap, temporal_split

log = get_logger("quality.audit")

# Ratios used only to audit outliers (Module 3 builds the full CAMELS set).
# Income items are YTD in Call Reports, so they are not used here.
AUDIT_RATIOS: dict[str, tuple[tuple[str, ...], pl.Expr]] = {
    "equity_to_assets": (("eq", "asset"), pl.col("eq") / pl.col("asset")),
    "npl_to_loans": (("nclnls", "lnlsgr"), pl.col("nclnls") / pl.col("lnlsgr")),
    "alll_to_loans": (("lnatres", "lnlsgr"), pl.col("lnatres") / pl.col("lnlsgr")),
    "loans_to_deposits": (("lnlsnet", "dep"), pl.col("lnlsnet") / pl.col("dep")),
    "securities_to_assets": (("sc", "asset"), pl.col("sc") / pl.col("asset")),
    "cash_to_assets": (("chbal", "asset"), pl.col("chbal") / pl.col("asset")),
    "brokered_to_deposits": (("bro", "dep"), pl.col("bro") / pl.col("dep")),
    "ore_to_assets": (("ore", "asset"), pl.col("ore") / pl.col("asset")),
    "log_assets": (("asset",), pl.col("asset").log1p()),
}

# (name, required columns, expression that is TRUE when the rule is violated)
DOMAIN_RULES: list[tuple[str, tuple[str, ...], pl.Expr]] = [
    ("total assets <= 0", ("asset",), pl.col("asset") <= 0),
    ("deposits > total assets", ("dep", "asset"), pl.col("dep") > pl.col("asset")),
    ("equity > total assets", ("eq", "asset"), pl.col("eq") > pl.col("asset")),
    ("negative equity (insolvent book value)", ("eq",), pl.col("eq") < 0),
    ("net loans > total assets", ("lnlsnet", "asset"), pl.col("lnlsnet") > pl.col("asset")),
    ("net loans > gross loans", ("lnlsnet", "lnlsgr"), pl.col("lnlsnet") > pl.col("lnlsgr") + 1),
    ("noncurrent loans > gross loans", ("nclnls", "lnlsgr"), pl.col("nclnls") > pl.col("lnlsgr")),
    ("allowance < 0", ("lnatres",), pl.col("lnatres") < 0),
    ("securities > total assets", ("sc", "asset"), pl.col("sc") > pl.col("asset")),
    ("|ROA| > 100%", ("roa",), pl.col("roa").abs() > 100),
    ("leverage ratio outside [-100, 100]%", ("rbc1aaj",), pl.col("rbc1aaj").abs() > 100),
]


def _safe_ratio_frame(df: pl.DataFrame) -> pl.DataFrame:
    exprs = [e.alias(n) for n, (req, e) in AUDIT_RATIOS.items() if set(req) <= set(df.columns)]
    out = df.select(*KEY_COLS, *[c for c in df.columns if c.startswith("target_")], *exprs)
    # x/0 -> inf and 0/0 -> NaN: both mean "undefined ratio" -> null. (Polars sorts NaN
    # above every number, so a NaN left in would poison mean/std and every comparison.)
    ratio_cols = [n for n in AUDIT_RATIOS if n in out.columns]
    return out.with_columns([pl.when(pl.col(c).is_infinite() | pl.col(c).is_nan())
                             .then(None).otherwise(pl.col(c)).alias(c) for c in ratio_cols])


def _numeric_value_cols(df: pl.DataFrame) -> list[str]:
    skip = set(KEY_COLS) | LEAKY_COLUMNS | META_COLS
    return [c for c, t in df.schema.items() if t.is_numeric() and c not in skip
            and not c.startswith("target_") and not c.startswith("_")]


# --------------------------------------------------------------------------- checks
def check_duplicates(paths: Paths, gold: pl.DataFrame) -> list[dict[str, Any]]:
    rows = []
    stats_file = paths.silver / "_silver_build_stats.json"
    if stats_file.exists():
        st = read_json(stats_file)
        for t in ("financials", "institutions", "failures"):
            s = st[t]
            rows.append({"table": t, "rows_bronze": s["rows_bronze"],
                         "exact_duplicates_removed": s["exact_duplicates_removed"],
                         "conflicting_key_duplicates": s.get("conflicting_key_duplicates", 0),
                         "rows_silver": s["rows_silver"],
                         "cast_failures": sum(s.get("cast_failures", {}).values())})
    dup_after = gold.select(KEY_COLS).is_duplicated().sum()
    rows.append({"table": "gold (re-check)", "rows_bronze": None, "exact_duplicates_removed": None,
                 "conflicting_key_duplicates": int(dup_after), "rows_silver": gold.height,
                 "cast_failures": None})
    return rows


def check_coverage(gold: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    per_q = gold.group_by("repdte").agg(banks=pl.col("cert").n_unique()).sort("repdte")
    qidx = (pl.col("repdte").dt.year() * 4 + pl.col("repdte").dt.quarter())
    gaps = (gold.select("cert", q=qidx).sort("cert", "q")
            .with_columns(step=pl.col("q").diff().over("cert"))
            .filter(pl.col("step") > 1))
    summary = {"quarters": per_q.height, "min_banks_per_quarter": int(per_q["banks"].min()),
               "max_banks_per_quarter": int(per_q["banks"].max()),
               "banks_with_internal_gaps": gaps["cert"].n_unique(),
               "missing_quarters_inside_series": int((gaps["step"] - 1).sum()) if gaps.height else 0}
    return per_q, summary


def profile_schema(gold: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for c, t in gold.schema.items():
        s = gold[c]
        rows.append({"column": c, "dtype": str(t), "null_pct": 100 * s.null_count() / len(s),
                     "n_unique": s.n_unique(),
                     "min": str(s.min()) if t.is_numeric() or t.is_temporal() else "",
                     "max": str(s.max()) if t.is_numeric() or t.is_temporal() else "",
                     "role": "LEAKY (never a feature)" if c in LEAKY_COLUMNS
                     else "key" if c in KEY_COLS else "target" if c.startswith("target_")
                     else "meta" if c in META_COLS or c.startswith("_") else "candidate"})
    return pl.DataFrame(rows)


def check_missingness(gold: pl.DataFrame, tgt: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    cols = _numeric_value_cols(gold)
    by_year = (gold.group_by("repyear")
               .agg([pl.col(c).is_null().mean().alias(c) for c in cols]).sort("repyear"))
    sized = gold.with_columns(
        size_q=(pl.col("asset").rank("ordinal").over("repdte") * 5
                / pl.len().over("repdte")).ceil().clip(1, 5).cast(pl.Int8))
    rows = []
    for c in cols:
        miss = pl.col(c).is_null()
        overall = float(gold[c].is_null().mean())
        yr = by_year[c]
        q = sized.group_by("size_q").agg(m=miss.mean())
        lab = sized.group_by(tgt).agg(m=miss.mean())
        lab_d = dict(zip(lab[tgt].to_list(), lab["m"].to_list()))
        pos, neg = lab_d.get(1, np.nan), lab_d.get(0, np.nan)
        q_gap = float(q["m"].max() - q["m"].min())
        lab_gap = float(abs(pos - neg)) if not (np.isnan(pos) or np.isnan(neg)) else np.nan
        structural = bool((yr.max() > 0.95) and (yr.min() < 0.05))
        if overall < 0.001:
            mech = "negligible (<0.1%)"
        elif structural:
            mech = "structural: item not reported in some years (form change)"
        elif not np.isnan(lab_gap) and lab_gap > 0.05:
            mech = "informative: differs failing vs non-failing -> keep missing flag (MNAR-like)"
        elif q_gap > 0.05:
            mech = "MAR: depends on bank size -> impute conditional / add flag"
        else:
            mech = "no association found (consistent with MCAR)"
        rows.append({"column": c, "missing_pct": 100 * overall,
                     "min_year_pct": 100 * float(yr.min()), "max_year_pct": 100 * float(yr.max()),
                     "size_quintile_gap_pp": 100 * q_gap,
                     "missing_pct_failing": 100 * pos if not np.isnan(pos) else None,
                     "missing_pct_nonfailing": 100 * neg if not np.isnan(neg) else None,
                     "evidence_based_mechanism": mech})
    return pl.DataFrame(rows).sort("missing_pct", descending=True), by_year


def check_variance(train: pl.DataFrame, full: pl.DataFrame, share: float) -> pl.DataFrame:
    rows = []
    for c in _numeric_value_cols(full):
        out = {"column": c}
        for name, df in (("train", train), ("full", full)):
            s = df[c].drop_nulls()
            if s.len() == 0:
                out[f"{name}_n_unique"], out[f"{name}_mode_share"], out[f"{name}_std"] = 0, 1.0, 0.0
                continue
            vc = s.value_counts(sort=True)
            out[f"{name}_n_unique"] = s.n_unique()
            out[f"{name}_mode_share"] = float(vc["count"][0] / s.len())
            out[f"{name}_std"] = float(s.std() or 0.0)
        out["drop_decision_train_only"] = (out["train_n_unique"] <= 1
                                           or out["train_mode_share"] >= share)
        rows.append(out)
    return pl.DataFrame(rows).sort("train_mode_share", descending=True)


def check_domain_rules(gold: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for name, req, expr in DOMAIN_RULES:
        if not set(req) <= set(gold.columns):
            continue
        v = gold.select(expr.fill_null(False).alias("v"))["v"]
        rows.append({"rule": name, "violations": int(v.sum()), "pct_rows": 100 * float(v.mean()),
                     "banks_affected": gold.filter(v)["cert"].n_unique()})
    return pl.DataFrame(rows)


def check_outliers(split, tgt: str, k: float) -> pl.DataFrame:
    """Fences are learned on TRAIN and applied unchanged to valid/test."""
    tr, va, te = (_safe_ratio_frame(d) for d in (split.train, split.valid, split.test))
    rows = []
    for c in [n for n in AUDIT_RATIOS if n in tr.columns]:
        s = tr[c].drop_nulls()
        if s.len() < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        lo_iqr, hi_iqr = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
        mu, sd = s.mean(), s.std()
        lo_g, hi_g = mu - 3 * sd, mu + 3 * sd
        r = {"ratio": c, "train_skew": float(s.skew()), "train_kurtosis": float(s.kurtosis()),
             "iqr_lower": lo_iqr, "iqr_upper": hi_iqr, "gauss_lower": lo_g, "gauss_upper": hi_g}
        for name, d in (("train", tr), ("valid", va), ("test", te)):
            x = d[c]
            iqr_flag = ((x < lo_iqr) | (x > hi_iqr)).fill_null(False)
            g_flag = ((x < lo_g) | (x > hi_g)).fill_null(False)
            r[f"{name}_iqr_flag_pct"] = 100 * float(iqr_flag.mean()) if d.height else None
            r[f"{name}_gauss_flag_pct"] = 100 * float(g_flag.mean()) if d.height else None
            if name == "train":
                y = d[tgt]
                r["fail_rate_pct_if_iqr_flagged"] = 100 * float(y.filter(iqr_flag).mean() or 0)
                r["fail_rate_pct_if_not_flagged"] = 100 * float(y.filter(~iqr_flag).mean() or 0)
        rows.append(r)
    return pl.DataFrame(rows)


def check_lof(split, tgt: str, cfg_q: dict[str, Any]) -> tuple[pl.DataFrame, pl.DataFrame]:
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler

    tr, va, te = (_safe_ratio_frame(d) for d in (split.train, split.valid, split.test))
    cols = [n for n in AUDIT_RATIOS if n in tr.columns]
    rng = np.random.default_rng(cfg_q["random_state"])
    n = min(tr.height, int(cfg_q["lof_max_train_rows"]))
    idx = np.sort(rng.choice(tr.height, size=n, replace=False))
    tr_s = tr[idx]
    prep = make_pipeline(SimpleImputer(strategy="median"), RobustScaler())
    Xtr = prep.fit_transform(tr_s.select(cols).to_numpy())  # fitted on the train sample only
    lof = LocalOutlierFactor(n_neighbors=int(cfg_q["lof_neighbors"]),
                             contamination=float(cfg_q["lof_contamination"]), novelty=True)
    lof.fit(Xtr)
    rows, top = [], None
    for name, d in (("train_sample", tr_s), ("valid", va), ("test", te)):
        if d.height == 0:
            continue
        X = prep.transform(d.select(cols).to_numpy())
        flag = lof.predict(X) == -1
        y = d[tgt].to_numpy()
        rows.append({"split": name, "rows": d.height, "lof_flag_pct": 100 * flag.mean(),
                     "fail_rate_pct_flagged": 100 * y[flag].mean() if flag.any() else None,
                     "fail_rate_pct_not_flagged": 100 * y[~flag].mean() if (~flag).any() else None})
        if name == "train_sample":
            score = -lof.score_samples(X)
            top = (d.select(KEY_COLS + [tgt] + cols).with_columns(lof_score=pl.Series(score))
                   .sort("lof_score", descending=True).head(20))
    return pl.DataFrame(rows), top


# --------------------------------------------------------------------------- figures
def _figures(paths: Paths, miss_by_year: pl.DataFrame, outl: pl.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cols = [c for c in miss_by_year.columns if c != "repyear"
            and miss_by_year[c].max() > 0]
    if cols:
        m = miss_by_year.select(cols).to_numpy().T * 100
        fig, ax = plt.subplots(figsize=(10, 0.28 * len(cols) + 1.5))
        im = ax.imshow(m, aspect="auto", cmap="Blues", vmin=0, vmax=100)
        ax.set_yticks(range(len(cols)), cols, fontsize=7)
        yrs = miss_by_year["repyear"].to_list()
        ax.set_xticks(range(len(yrs)), yrs, rotation=90, fontsize=7)
        ax.set_title("Missing values by report year (% of bank-quarters)")
        fig.colorbar(im, ax=ax, fraction=0.025)
        fig.tight_layout()
        fig.savefig(paths.figures / "dq_missingness_by_year.png", dpi=200)
        plt.close(fig)
    if outl.height:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        y = np.arange(outl.height)
        ax.barh(y - 0.2, outl["train_iqr_flag_pct"].to_list(), 0.4, label="Tukey IQR (k=3)")
        ax.barh(y + 0.2, outl["train_gauss_flag_pct"].to_list(), 0.4, label="mean ± 3 sd")
        ax.set_yticks(y, outl["ratio"].to_list(), fontsize=8)
        ax.set_xlabel("% of train bank-quarters flagged")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(paths.figures / "dq_outlier_rules.png", dpi=200)
        plt.close(fig)


# --------------------------------------------------------------------------- driver
def run_audit(cfg: dict[str, Any]) -> dict[str, Any]:
    paths: Paths = cfg["_paths"]
    h = int(cfg["gold"]["label_horizon_quarters"])
    tgt = target_name(h)
    q = cfg["quality"]
    gold = read_gold(cfg, lazy=False)
    split = temporal_split(gold, cfg)
    assert_no_temporal_overlap(split, h)
    T = paths.tables

    dup = check_duplicates(paths, gold)
    per_q, cov = check_coverage(gold)
    prof = profile_schema(gold)
    miss, miss_by_year = check_missingness(gold, tgt)
    var = check_variance(split.train, gold, float(q["near_constant_share"]))
    rules = check_domain_rules(gold)
    outl = check_outliers(split, tgt, float(q["iqr_k"]))
    lof, lof_top = check_lof(split, tgt, q)
    split_tbl = pl.DataFrame(split.stats)

    pl.DataFrame(dup).write_csv(T / "dq_duplicates.csv")
    per_q.write_csv(T / "dq_banks_per_quarter.csv")
    prof.write_csv(T / "dq_schema_profile.csv")
    miss.write_csv(T / "dq_missingness.csv")
    miss_by_year.write_csv(T / "dq_missingness_by_year.csv")
    var.write_csv(T / "dq_variance_train_only.csv")
    rules.write_csv(T / "dq_domain_rules.csv")
    outl.write_csv(T / "dq_outliers_iqr_vs_gauss.csv")
    lof.write_csv(T / "dq_lof_summary.csv")
    if lof_top is not None:
        lof_top.write_csv(T / "dq_lof_top20_train.csv")
    split_tbl.write_csv(T / "dq_split_summary.csv")
    _figures(paths, miss_by_year, outl)

    dropped = var.filter(pl.col("drop_decision_train_only"))["column"].to_list()
    md = [f"# Module 2 - Data-quality audit (measured, generated {utc_now()})\n",
          "All numbers below are computed by `python pipeline.py audit` on the data in "
          f"`{paths.data}`. Fences, variance filters and LOF are fitted on the TRAIN window only.\n",
          "## 1. Duplicates and lineage\n", md_table(dup),
          "## 2. Panel coverage\n", md_table([cov]),
          "## 3. Temporal split (purged) and class balance\n", md_table(split.stats),
          "## 4. Missingness and evidence for its mechanism\n",
          "_MNAR cannot be proven from observed data; the label below is a "
          "data-driven indication only._\n",
          md_table(miss.filter(pl.col("missing_pct") > 0).to_dicts()),
          "## 5. Zero / near-zero variance (decided on train)\n",
          f"Columns dropped: {dropped or 'none'}\n\n",
          md_table(var.head(15).to_dicts()),
          "## 6. Accounting identity and domain-rule violations\n", md_table(rules.to_dicts()),
          "## 7. Univariate outliers: Tukey IQR vs mean ± 3 sd (fences fitted on train)\n",
          md_table(outl.select("ratio", "train_skew", "train_kurtosis", "train_iqr_flag_pct",
                               "train_gauss_flag_pct", "test_iqr_flag_pct",
                               "fail_rate_pct_if_iqr_flagged",
                               "fail_rate_pct_if_not_flagged").to_dicts()),
          "## 8. Multivariate outliers: Local Outlier Factor (novelty mode)\n", md_table(lof.to_dicts())]
    (paths.reports / "module2_data_quality.md").write_text("\n".join(md), encoding="utf-8")
    log.info("audit written to %s", paths.reports / "module2_data_quality.md")
    return {"coverage": cov, "split": split.stats, "dropped_near_constant": dropped}
