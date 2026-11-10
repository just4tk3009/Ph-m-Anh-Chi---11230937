"""Logic checks on the pipeline's real outputs (``python pipeline.py check``).

Unit tests prove the code on synthetic data; these checks re-derive key facts
from the real Silver/Gold/feature tables and from the written reports, so a
logic error in the full run (a wrong join, a label off by one quarter, a split
that overlaps, a report number that does not match the data) shows up as a
failed row in ``reports/tables/sanity_checks.csv``. Nothing is fitted here.
"""
from __future__ import annotations

import json
from datetime import timedelta
from typing import Any, Callable

import polars as pl

from ..storage.gold import LEAKY_COLUMNS, read_gold, target_name
from ..utils.config import Paths, to_date
from ..utils.io_utils import md_table, utc_now
from ..utils.logging_utils import get_logger
from .temporal_split import assert_no_temporal_overlap, temporal_split

log = get_logger("quality.sanity")


def run_checks(cfg: dict[str, Any]) -> dict[str, Any]:
    paths: Paths = cfg["_paths"]
    h = int(cfg["gold"]["label_horizon_quarters"])
    tgt = target_name(h)
    T = paths.tables
    rows: list[dict[str, Any]] = []

    def check(area: str, name: str, fn: Callable[[], tuple[bool, Any]]) -> None:
        try:
            ok, detail = fn()
        except Exception as e:  # a crash is a failed check, not a crashed run
            ok, detail = False, f"error: {type(e).__name__}: {e}"
        rows.append({"area": area, "check": name, "passed": bool(ok), "detail": str(detail)})
        log.info("%-5s %s / %s: %s", "OK" if ok else "FAIL", area, name, detail)

    silver = pl.read_parquet(paths.silver / "financials.parquet", columns=["cert", "repdte"])
    fails = pl.read_parquet(paths.silver / "failures.parquet")
    gold = read_gold(cfg, lazy=False)
    feats = pl.read_parquet(paths.gold / "features.parquet")
    s = cfg["split"]

    # ---- Part 1: completeness and keys
    check("storage", "silver (cert, repdte) unique",
          lambda: (silver.select(["cert", "repdte"]).is_duplicated().sum() == 0, silver.height))
    check("storage", "every quarter 1984Q1..2026Q2 present",
          lambda: (silver["repdte"].n_unique() == 170, silver["repdte"].n_unique()))
    check("storage", "failures: one row per cert, only RESTYPE=FAILURE",
          lambda: (fails["cert"].is_duplicated().sum() == 0
                   and set(fails["restype"].drop_nulls().unique()) <= {"FAILURE"},
                   fails["restype"].value_counts().to_dicts()))

    # ---- Part 2: label and panel logic
    pos = gold.filter(pl.col(tgt) == 1)
    check("label", "positives: failure strictly after the report date",
          lambda: ((pos["faildate"] > pos["repdte"]).all(), pos.height))
    check("label", "positives: failure within 4 quarters (1..4 quarter-ends ahead)",
          lambda: (pos["quarters_to_failure"].is_between(1, h).all(),
                   pos["quarters_to_failure"].value_counts().sort("quarters_to_failure").to_dicts()))
    neg_failed = gold.filter((pl.col(tgt) == 0) & pl.col("faildate").is_not_null()
                             & (pl.col("quarters_to_failure").is_between(1, h)))
    check("label", "no negative row 1..4 quarter-ends before a failure",
          lambda: (neg_failed.height == 0, neg_failed.height))
    check("label", "no report on/after the failure date kept",
          lambda: ((gold.filter(pl.col("faildate").is_not_null()
                                & (pl.col("repdte") >= pl.col("faildate"))).height == 0), "0 rows"))
    per_bank = pos.group_by("cert").len()["len"]
    check("label", "at most 4 positive quarters per failed bank",
          lambda: (per_bank.max() <= h, f"mean {per_bank.mean():.2f}, max {per_bank.max()}"))
    gp = gold.filter(pl.col("faildate").is_not_null()).group_by("cert").agg(
        last=pl.col("repdte").max(), fd=pl.col("faildate").first())
    late = gp.filter((pl.col("fd").dt.year() * 4 + pl.col("fd").dt.quarter())
                     - (pl.col("last").dt.year() * 4 + pl.col("last").dt.quarter()) > h)
    check("label", "failed banks whose last report is > 4 quarters before failure",
          lambda: (True, f"{late.height} banks (they get no positive row; reported, not an error)"))
    in_list = fails.filter(pl.col("faildate") > to_date(s["train_start"]) - timedelta(days=90))
    matched = in_list.join(gold.select("cert").unique(), on="cert", how="semi").height
    check("label", "failures after 1984Q1 found in the panel",
          lambda: (matched / max(1, in_list.height) > 0.99, f"{matched}/{in_list.height}"))
    merged_pos = gold.filter((pl.col(tgt) == 1) & pl.col("faildate").is_null()).height
    check("label", "no positive without a failure date (mergers are 0)",
          lambda: (merged_pos == 0, merged_pos))
    check("panel", "gold (cert, repdte) unique",
          lambda: (gold.select(["cert", "repdte"]).is_duplicated().sum() == 0, gold.height))
    check("panel", "available_date = repdte + publication lag",
          lambda: (((gold["available_date"] - gold["repdte"]).dt.total_days()
                    == int(cfg["gold"]["publication_lag_days"])).all(), "all rows"))
    early = pos.filter(pl.col("faildate") <= pl.col("available_date"))
    check("panel", "positives whose failure precedes the report's publication",
          lambda: (True, f"{early.height} of {pos.height} positive rows "
                         f"({100 * early.height / max(1, pos.height):.1f}%): the report was "
                         "not yet public when the bank failed (limitation, not an error)"))

    # ---- split
    split = temporal_split(gold, cfg)
    check("split", "purged: no label window crosses into the next split",
          lambda: (assert_no_temporal_overlap(split, h) is None, "train->valid, valid->test"))
    gap_q = ((to_date(s["test_start"]).year * 4 + (to_date(s["test_start"]).month - 1) // 3)
             - (to_date(s["valid_end"]).year * 4 + (to_date(s["valid_end"]).month - 1) // 3) - 1)
    check("split", "gap development -> test >= 4 quarters", lambda: (gap_q >= h, f"{gap_q} quarters"))
    tf = split.test.filter(pl.col(tgt) == 1)["cert"].n_unique()
    check("split", "test has >= 50 failed banks", lambda: (tf >= 50, tf))
    check("split", "test ends at 2025Q2 (complete labels)",
          lambda: (str(split.test["repdte"].max()) <= "2025-06-30", str(split.test["repdte"].max())))

    # ---- features
    fcols = [c for c in feats.columns if c not in LEAKY_COLUMNS and not c.startswith("target_")]
    check("features", "features.parquet has one row per gold row",
          lambda: (feats.height == gold.height, f"{feats.height} vs {gold.height}"))
    sel = json.loads((paths.gold / "selected_features.json").read_text(encoding="utf-8"))
    bad = [f for f in sel["selected_features"] if f in LEAKY_COLUMNS or f.startswith("target_")]
    check("features", "selected features contain no leaky column", lambda: (not bad, bad or "none"))
    inf_cols = [c for c in fcols if feats.schema[c].is_float()
                and feats.select(pl.col(c).is_infinite().any()).item()]
    check("features", "no +/-inf in any feature", lambda: (not inf_cols, inf_cols or "none"))
    lk = pl.read_csv(T / "features_lookahead_check.csv")
    check("features", "look-ahead check passed", lambda: (bool(lk["passed"][0]), lk.to_dicts()[0]))

    # ---- model and reports agree with the data
    if (T / "model_choice.json").exists():
        looks_f = paths.gold / "_test_evaluations.jsonl"
        looks = looks_f.read_text(encoding="utf-8").splitlines() if looks_f.exists() else []
        check("model", "test window evaluated exactly once",
              lambda: (len([x for x in looks if x.strip()]) == 1, f"{len(looks)} evaluation(s)"))
        tt = pl.read_csv(T / "model_test_results.csv")
        check("model", "test positives in results = positives in the test split",
              lambda: ((tt["positives"] == int(split.test[tgt].sum())).all(),
                       f"{tt['positives'][0]} vs {int(split.test[tgt].sum())}"))
        ch = json.loads((T / "model_choice.json").read_text(encoding="utf-8"))
        cv = pl.read_csv(T / "model_cv_summary.csv")
        best = cv.filter(~pl.col("model").str.starts_with("baseline")).sort(
            "pr_auc_mean", descending=True)["model"][0]
        check("model", "chosen model = best non-baseline mean CV PR-AUC",
              lambda: (ch["chosen_model"] == best, f"{ch['chosen_model']} vs {best}"))
    facts_f = T / "report_page1_facts.json"
    if facts_f.exists():
        fa = json.loads(facts_f.read_text(encoding="utf-8"))
        check("report", "page-1 test failed banks = split", lambda: (fa["test_failed_banks"] == tf,
                                                                    f"{fa['test_failed_banks']} vs {tf}"))
        check("report", "page-1 financial rows = silver rows",
              lambda: (fa["financials_rows"] == silver.height,
                       f"{fa['financials_rows']} vs {silver.height}"))

    tbl = pl.DataFrame(rows)
    tbl.write_csv(T / "sanity_checks.csv")
    n_fail = int((~tbl["passed"]).sum())
    (T / "sanity_checks.md").write_text(
        f"# Logic checks on the real outputs (generated {utc_now()})\n\n"
        f"{tbl.height} checks, **{n_fail} failed**.\n\n" + md_table(tbl.to_dicts()),
        encoding="utf-8")
    return {"checks": tbl.height, "failed": n_fail,
            "failed_checks": tbl.filter(~pl.col("passed"))["check"].to_list()}
