"""Tests for Module 3: de-cumulation, look-ahead safety, engine equivalence."""
from datetime import date

import polars as pl
import pytest

from fdic_ews.features import engines as E
from fdic_ews.features.camels import (add_quarterly_flows, build_features_lazy,
                                      verify_no_lookahead)
from fdic_ews.features.processing_benchmark import _compare
from fdic_ews.storage.gold import read_gold


def test_ytd_decumulation_is_gap_aware():
    df = pl.DataFrame({
        "cert": [1, 1, 1, 1, 1],
        "repdte": [date(2020, 3, 31), date(2020, 6, 30), date(2020, 12, 31),
                   date(2021, 3, 31), date(2021, 6, 30)],
        "netinc": [10.0, 25.0, 60.0, 5.0, 12.0]})
    out = add_quarterly_flows(df.lazy()).collect()["netinc_q"].to_list()
    # Q1 = YTD; Q2 = 25-10; Q4 follows a missing Q3 -> unknown; next-year Q1 = YTD; Q2 = 12-5
    assert out == [10.0, 15.0, None, 5.0, 7.0]


def test_lookahead_check_passes(synthetic_cfg):
    gold = read_gold(synthetic_cfg, lazy=False)
    res = verify_no_lookahead(gold, date(2010, 12, 31))
    assert res["passed"] and res["rows_compared"] > 0


def test_lookahead_check_detects_a_leaky_feature(synthetic_cfg, monkeypatch):
    """The check must have power: a feature using t+1 information must be caught."""
    import fdic_ews.features.camels as C
    original = C.add_dynamic_features

    def leaky(lf, base):
        lf = original(lf, base)
        return lf.with_columns(pl.col("npl_ratio").shift(-1).over("cert").alias("npl_ratio_ma4"))

    monkeypatch.setattr(C, "add_dynamic_features", leaky)
    gold = read_gold(synthetic_cfg, lazy=False)
    res = C.verify_no_lookahead(gold, date(2010, 12, 31))
    assert not res["passed"] and "npl_ratio_ma4" in res["features_changed_by_future_rows"]


def test_features_have_no_leaky_columns_as_features(synthetic_cfg):
    from fdic_ews.features.camels import feature_names
    from fdic_ews.quality.preprocessing import assert_no_leaky_columns
    assert_no_leaky_columns(feature_names())
    cols = build_features_lazy(read_gold(synthetic_cfg)).collect_schema().names()
    assert "target_fail_4q" in cols and "cert" in cols


@pytest.mark.parametrize("engine", ["pandas", "polars_eager"])
def test_engines_agree(synthetic_cfg, engine):
    path = str(synthetic_cfg["_paths"].silver / "financials.parquet")
    ref = E.run_polars_lazy(path)
    cmp = _compare(ref, E.ENGINES[engine](path))
    assert cmp["count_mismatches"] == 0 and cmp["max_rel_diff_sums"] < 1e-9


def test_spark_agrees(synthetic_cfg):
    pytest.importorskip("pyspark")
    path = str(synthetic_cfg["_paths"].silver / "financials.parquet")
    try:
        res = E.run_spark(path)
    except Exception as err:  # no Java on this machine
        pytest.skip(f"Spark unavailable: {err}")
    cmp = _compare(E.run_polars_lazy(path), res)
    assert cmp["count_mismatches"] == 0 and cmp["max_rel_diff_sums"] < 1e-9


def test_rolling_window_is_gap_aware_and_ignores_nulls():
    from fdic_ews.features.camels import add_dynamic_features, add_quarterly_flows
    d = [date(2020, 3, 31), date(2020, 6, 30), date(2020, 9, 30), date(2020, 12, 31),
         date(2021, 6, 30), date(2021, 9, 30)]
    df = pl.DataFrame({"cert": [1] * 6, "repdte": d, "x": [1.0, 2.0, 3.0, 4.0, 5.0, None],
                       "asset": [1.0] * 6})
    lf = add_quarterly_flows(df.lazy())
    out = add_dynamic_features(lf, ["x"]).sort("repdte").collect()
    # 2021Q2 window = 2020Q3..2021Q2 -> {3, 4, 5} (2021Q1 missing, 2020Q2 outside)
    assert out["x_ma4"].to_list() == [1.0, 1.5, 2.0, 2.5, 4.0, 4.5]
    assert out["x_sd4"][0] is None and abs(out["x_sd4"][4] - 1.0) < 1e-12
    assert out["x_qoq"].to_list() == [None, 1.0, 1.0, 1.0, None, None]
