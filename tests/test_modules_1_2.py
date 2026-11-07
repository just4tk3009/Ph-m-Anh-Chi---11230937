"""Unit + smoke tests for Modules 1-2 (run: pytest -q)."""
import json

import numpy as np
import pandas as pd
import polars as pl
import pytest

from fdic_ews.ingestion.fdic_client import FDICClient
from fdic_ews.quality.preprocessing import (IQRClipper, LeakageError, NearConstantDropper,
                                            assert_no_leaky_columns)
from fdic_ews.quality.temporal_split import assert_no_temporal_overlap, temporal_split
from fdic_ews.storage.gold import read_gold, target_name
from fdic_ews.utils.io_utils import quarter_ends


def test_quarter_calendar():
    q = quarter_ends("2008Q3", "2009Q2")
    assert [d.isoformat() for d in q] == ["2008-09-30", "2008-12-31", "2009-03-31", "2009-06-30"]


def test_parse_fdic_envelope():
    raw = json.dumps({"meta": {"total": 2}, "data": [{"data": {"CERT": 1}}, {"data": {"CERT": 2}}]})
    rows, total = FDICClient.parse(raw.encode())
    assert total == 2 and rows == [{"CERT": 1}, {"CERT": 2}]


def test_iqr_clipper_uses_train_only():
    train = pd.DataFrame({"x": np.arange(100, dtype=float)})
    test = pd.DataFrame({"x": [1e9, -1e9, 50.0]})
    clip = IQRClipper(k=3).fit(train)
    before = (clip.lower_.copy(), clip.upper_.copy())
    out = clip.transform(test)
    assert out[0, 0] == clip.upper_[0] and out[1, 0] == clip.lower_[0] and out[2, 0] == 50
    clip.transform(test * 10)  # transforming never refits
    assert np.array_equal(before[0], clip.lower_) and np.array_equal(before[1], clip.upper_)


def test_near_constant_dropper():
    X = pd.DataFrame({"const": [1.0] * 1000, "almost": [0.0] * 998 + [1.0, 2.0],
                      "ok": np.random.default_rng(0).normal(size=1000), "empty": [np.nan] * 1000})
    d = NearConstantDropper(0.995).fit(X)
    assert d.keep_ == ["ok"] and set(d.dropped_) == {"const", "almost", "empty"}


def test_leaky_columns_rejected():
    with pytest.raises(LeakageError):
        assert_no_leaky_columns(["asset", "faildate"])
    with pytest.raises(LeakageError):
        assert_no_leaky_columns(["asset", "target_fail_4q"])
    assert_no_leaky_columns(["asset", "eq"])


def test_gold_label_definition(synthetic_cfg):
    g = read_gold(synthetic_cfg, lazy=False)
    tgt = target_name(4)
    pos = g.filter(pl.col(tgt) == 1)
    assert pos.height > 0
    assert (pos["faildate"] > pos["repdte"]).all()
    assert (pos["quarters_to_failure"] <= 4).all()
    assert g.filter(pl.col("faildate").is_not_null() & (pl.col("repdte") >= pl.col("faildate"))).height == 0
    assert g.select(["cert", "repdte"]).is_duplicated().sum() == 0
    # an open-bank assistance record must not remove the assisted bank's later reports
    from fdic_ews.utils.io_utils import read_json
    stats = read_json(synthetic_cfg["_paths"].gold / "_gold_build_stats.json")
    assert stats["post_failure_rows_dropped"] == 0


def test_split_is_purged_and_ordered(synthetic_cfg):
    g = read_gold(synthetic_cfg, lazy=False)
    s = temporal_split(g, synthetic_cfg)
    assert_no_temporal_overlap(s, 4)
    st = {r["split"]: r for r in s.stats}
    assert st["train"]["rows_purged"] > 0          # train labels may not reach the valid period
    # valid labels may use the >= 4-quarter gap but never the test period
    assert st["valid"]["max_label_window_end"] < st["test"]["from"]
    assert s.test["repdte"].min().isoformat() >= str(synthetic_cfg["split"]["test_start"])


def test_audit_and_leakage_run(synthetic_cfg):
    from fdic_ews.quality.audit import run_audit
    from fdic_ews.quality.leakage_checks import run_leakage_checks
    run_audit(synthetic_cfg)
    summary = run_leakage_checks(synthetic_cfg)
    assert summary["all_medians_equal_train_median"]
    assert summary["params_invariant_to_test_data"]
    assert (synthetic_cfg["_paths"].reports / "module2_data_quality.md").exists()


def test_silver_keeps_text_and_date_columns(synthetic_cfg):
    """Regression: STNAME was cast to Float64 (every value lost) and INSDATE not parsed."""
    from fdic_ews.utils.io_utils import read_json
    paths = synthetic_cfg["_paths"]
    st = read_json(paths.silver / "_silver_build_stats.json")
    for t in ("financials", "institutions"):
        assert not st[t].get("cast_failures"), st[t].get("cast_failures")
    inst = pl.read_parquet(paths.silver / "institutions.parquet")
    assert inst["stname"].dtype == pl.String and inst["stname"].null_count() == 0
    assert inst["insdate"].dtype == pl.Date and inst["insdate"].null_count() == 0


def test_structural_zero_detected_and_treated_as_missing(synthetic_cfg):
    from fdic_ews.quality.audit import check_zero_share
    g = read_gold(synthetic_cfg, lazy=False)
    summ, _ = check_zero_share(g)
    flagged = set(summ.filter(pl.col("structural_zero"))["column"])
    assert {"rbc1aaj", "rbcrwaj"} <= flagged
    assert "asset" not in flagged
